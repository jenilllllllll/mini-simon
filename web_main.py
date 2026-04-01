"""FastAPI backend for Mini-Simon Indian Market Dashboard.
Provides REST endpoints and HTML dashboard for live trading data.
"""

from __future__ import annotations

import os
import queue
import threading
import asyncio
import logging
import time
import json
import io
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time as dt_time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pytz
from fastapi import FastAPI, Request, Body, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from pydantic import BaseModel

from fyers_apiv3 import fyersModel

try:  # Fyers v3 WebSocket SDK (used for live ticks on the dashboard)
    from fyers_apiv3.FyersWebsocket import data_ws  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency / defensive
    data_ws = None  # type: ignore[assignment]

from config import get_config
from live_engine import EngineManager
from live_data_feed import LiveCandleManager
from logger_config import LoggerConfig
from live_signal_aggregator import AggregatedSignal
from mcx_symbols import build_commodity_symbol_mapping, get_current_commodity_symbol, get_mcx_contract_meta
from live_strategy_runner import LiveStrategyRunner

# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------

app = FastAPI(title="Mini-Simon Indian Markets Dashboard", version="1.0.0")

# Mount static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

logger = logging.getLogger("mini_simon_dashboard")

# -----------------------------------------------------------------------------
# Global state and locks
# -----------------------------------------------------------------------------

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class _SmcDecision:
    valid: bool
    delay: bool
    reason: str
    order_type: str
    limit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]


def _tf_minutes(tf: str) -> int:
    t = str(tf or "").strip().lower()
    if t.endswith("m"):
        try:
            return int(t[:-1])
        except Exception:
            return 0
    if t.endswith("h"):
        try:
            return int(t[:-1]) * 60
        except Exception:
            return 0
    if t in {"1d", "d", "day"}:
        return 60 * 24
    try:
        return int(t)
    except Exception:
        return 0


def _find_recent_swings(df: pd.DataFrame, lookback: int = 3) -> Tuple[Optional[float], Optional[float]]:
    if df is None or df.empty or len(df) < (lookback * 2 + 3):
        return None, None
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    for i in range(lookback, len(df) - lookback):
        h = float(highs[i])
        l = float(lows[i])
        is_sh = True
        is_sl = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if float(highs[j]) >= h:
                is_sh = False
            if float(lows[j]) <= l:
                is_sl = False
            if not is_sh and not is_sl:
                break
        if is_sh:
            swing_highs.append(h)
        if is_sl:
            swing_lows.append(l)
    return (swing_highs[-1] if swing_highs else None), (swing_lows[-1] if swing_lows else None)


def _equal_level(values: List[float], rel_tol: float = 0.0008) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    v_sorted = sorted([float(v) for v in values if v is not None])
    if len(v_sorted) < 2:
        return None
    clusters: List[List[float]] = []
    for v in v_sorted:
        placed = False
        for c in clusters:
            ref = float(sum(c) / max(len(c), 1))
            if ref > 0 and abs(v - ref) / ref <= rel_tol:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    clusters = [c for c in clusters if len(c) >= 2]
    if not clusters:
        return None
    best = max(clusters, key=lambda c: len(c))
    return float(sum(best) / len(best))


def _detect_fvg_and_ob(df: pd.DataFrame) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if df is None or df.empty or len(df) < 5:
        return None, None

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    fvg: Optional[Tuple[float, float]] = None
    for i in range(len(df) - 1, 1, -1):
        if float(lows[i]) > float(highs[i - 2]):
            fvg = (float(highs[i - 2]), float(lows[i]))
            break
        if float(highs[i]) < float(lows[i - 2]):
            fvg = (float(highs[i]), float(lows[i - 2]))
            break

    demand_ob: Optional[Tuple[float, float]] = None
    supply_ob: Optional[Tuple[float, float]] = None
    for i in range(len(df) - 3, 1, -1):
        prev_bear = float(closes[i]) < float(opens[i])
        prev_bull = float(closes[i]) > float(opens[i])
        if prev_bear:
            if float(closes[i + 1]) > float(highs[i]) and float(closes[i + 2]) > float(highs[i]):
                demand_ob = (float(lows[i]), float(highs[i]))
                break
        if prev_bull:
            if float(closes[i + 1]) < float(lows[i]) and float(closes[i + 2]) < float(lows[i]):
                supply_ob = (float(lows[i]), float(highs[i]))
                break

    return fvg, (demand_ob or supply_ob)


def is_smc_valid(
    symbol_code: str,
    direction: str,
    entry_price: float,
    ltf_df: Optional[pd.DataFrame] = None,
    htf_df: Optional[pd.DataFrame] = None,
    ltf_timeframe: str = "5m",
    htf_timeframe: str = "60m",
) -> _SmcDecision:
    dir_up = str(direction or "").upper()
    if dir_up not in {"BUY", "SELL"}:
        return _SmcDecision(False, False, "InvalidDirection", "MARKET", None, None, None)

    try:
        px = float(entry_price)
    except Exception:
        return _SmcDecision(False, False, "InvalidEntryPrice", "MARKET", None, None, None)
    if px <= 0:
        return _SmcDecision(False, False, "InvalidEntryPrice", "MARKET", None, None, None)

    if ltf_df is None or ltf_df.empty:
        ltf_df = _get_historical_candles(symbol_code, ltf_timeframe, 220)
    if htf_df is None or htf_df.empty:
        htf_df = _get_historical_candles(symbol_code, htf_timeframe, 220)

    if ltf_df is None or ltf_df.empty or htf_df is None or htf_df.empty:
        return _SmcDecision(True, False, "NoCandleContext", "MARKET", None, None, None)

    htf_swing_high, htf_swing_low = _find_recent_swings(htf_df, lookback=3)
    if htf_swing_high is None or htf_swing_low is None or htf_swing_high == htf_swing_low:
        return _SmcDecision(True, False, "NoHTFSwings", "MARKET", None, None, None)

    if dir_up == "BUY":
        swing_low = float(min(htf_swing_low, htf_swing_high))
        swing_high = float(max(htf_swing_low, htf_swing_high))
        fib50 = swing_low + 0.5 * (swing_high - swing_low)
        fib786 = swing_low + 0.786 * (swing_high - swing_low)
        if px > fib50:
            return _SmcDecision(False, False, "PremiumZone", "MARKET", None, None, None)
        prefer_discount = abs(px - fib786) / max(px, 1e-9) <= 0.004

        recent = ltf_df.tail(120).copy()
        lows = recent["low"].to_numpy(dtype=float)
        swing_lows: List[float] = []
        for i in range(3, len(recent) - 3):
            l = float(lows[i])
            if all(float(lows[j]) > l for j in range(i - 3, i + 4) if j != i):
                swing_lows.append(l)
        eq_low = _equal_level(swing_lows[-12:], rel_tol=0.0009)
        if eq_low is not None and len(recent) >= 3:
            lo0 = float(recent["low"].iloc[-1])
            lo1 = float(recent["low"].iloc[-2])
            cl0 = float(recent["close"].iloc[-1])
            cl1 = float(recent["close"].iloc[-2])
            swept = (lo1 < eq_low) or (lo0 < eq_low)
            reclaimed = (cl0 > eq_low) or (cl1 > eq_low)
            if not swept or not reclaimed:
                return _SmcDecision(False, True, "WaitSweepReclaim_EQ_Low", "MARKET", None, None, None)

        fvg, ob = _detect_fvg_and_ob(recent)
        order_type = "MARKET"
        limit_price: Optional[float] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None

        if ob is not None:
            ob_low, ob_high = float(min(ob[0], ob[1])), float(max(ob[0], ob[1]))
            stop_loss = ob_low * (1.0 - 0.001)
        if fvg is not None:
            f_low, f_high = float(min(fvg[0], fvg[1])), float(max(fvg[0], fvg[1]))
            inside_fvg = (px >= f_low) and (px <= f_high)
            if inside_fvg and ob is not None:
                ob_low, ob_high = float(min(ob[0], ob[1])), float(max(ob[0], ob[1]))
                if ob_high < px:
                    order_type = "LIMIT"
                    limit_price = ob_high

        highs = recent["high"].to_numpy(dtype=float)
        swing_highs: List[float] = []
        for i in range(3, len(recent) - 3):
            h = float(highs[i])
            if all(float(highs[j]) < h for j in range(i - 3, i + 4) if j != i):
                swing_highs.append(h)
        eq_high = _equal_level(swing_highs[-12:], rel_tol=0.0009)
        if eq_high is not None:
            take_profit = eq_high
        else:
            take_profit = float(htf_swing_high)

        rsn = "OK"
        if prefer_discount:
            rsn = "OK_Fib786"
        return _SmcDecision(True, False, rsn, order_type, limit_price, stop_loss, take_profit)

    swing_high = float(max(htf_swing_low, htf_swing_high))
    swing_low = float(min(htf_swing_low, htf_swing_high))
    fib50 = swing_high - 0.5 * (swing_high - swing_low)
    fib786 = swing_high - 0.786 * (swing_high - swing_low)
    if px < fib50:
        return _SmcDecision(False, False, "DiscountZone_SellDisabled", "MARKET", None, None, None)
    prefer_premium = abs(px - fib786) / max(px, 1e-9) <= 0.004

    recent = ltf_df.tail(120).copy()
    highs = recent["high"].to_numpy(dtype=float)
    swing_highs2: List[float] = []
    for i in range(3, len(recent) - 3):
        h = float(highs[i])
        if all(float(highs[j]) < h for j in range(i - 3, i + 4) if j != i):
            swing_highs2.append(h)
    eq_high2 = _equal_level(swing_highs2[-12:], rel_tol=0.0009)
    if eq_high2 is not None and len(recent) >= 3:
        hi0 = float(recent["high"].iloc[-1])
        hi1 = float(recent["high"].iloc[-2])
        cl0 = float(recent["close"].iloc[-1])
        cl1 = float(recent["close"].iloc[-2])
        swept = (hi1 > eq_high2) or (hi0 > eq_high2)
        reclaimed = (cl0 < eq_high2) or (cl1 < eq_high2)
        if not swept or not reclaimed:
            return _SmcDecision(False, True, "WaitSweepReclaim_EQ_High", "MARKET", None, None, None)

    fvg2, ob2 = _detect_fvg_and_ob(recent)
    order_type2 = "MARKET"
    limit_price2: Optional[float] = None
    stop_loss2: Optional[float] = None
    take_profit2: Optional[float] = None

    if ob2 is not None:
        ob_low, ob_high = float(min(ob2[0], ob2[1])), float(max(ob2[0], ob2[1]))
        stop_loss2 = ob_high * (1.0 + 0.001)
    if fvg2 is not None:
        f_low, f_high = float(min(fvg2[0], fvg2[1])), float(max(fvg2[0], fvg2[1]))
        inside_fvg = (px >= f_low) and (px <= f_high)
        if inside_fvg and ob2 is not None:
            ob_low, ob_high = float(min(ob2[0], ob2[1])), float(max(ob2[0], ob2[1]))
            if ob_low > px:
                order_type2 = "LIMIT"
                limit_price2 = ob_low

    lows2 = recent["low"].to_numpy(dtype=float)
    swing_lows2: List[float] = []
    for i in range(3, len(recent) - 3):
        l = float(lows2[i])
        if all(float(lows2[j]) > l for j in range(i - 3, i + 4) if j != i):
            swing_lows2.append(l)
    eq_low2 = _equal_level(swing_lows2[-12:], rel_tol=0.0009)
    if eq_low2 is not None:
        take_profit2 = eq_low2
    else:
        take_profit2 = float(htf_swing_low)

    rsn2 = "OK"
    if prefer_premium:
        rsn2 = "OK_Fib786"
    return _SmcDecision(True, False, rsn2, order_type2, limit_price2, stop_loss2, take_profit2)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _infer_session_label(ts: datetime) -> str:
    h = ts.hour + (ts.minute / 60.0)
    # Rough FX-style sessions in IST; good enough for filtering.
    if 5.5 <= h < 12.0:
        return "asia"
    if 12.0 <= h < 15.5:
        return "europe"
    if 15.5 <= h < 18.0:
        return "overlap"
    if 18.0 <= h < 23.0:
        return "us"
    return "off_hours"


def _parse_dt_best_effort(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        try:
            return IST.localize(dt)
        except Exception:
            return dt
    try:
        return dt.astimezone(IST)
    except Exception:
        return dt


def _trade_to_pure_schema(trade: Dict[str, Any], idx: int) -> Dict[str, Any]:
    trade_id = trade.get("trade_id")
    if trade_id is None:
        trade_id = trade.get("id")
    if trade_id is None:
        trade_id = idx + 1

    symbol = (
        trade.get("display_symbol")
        or trade.get("instrument")
        or trade.get("symbol")
        or ""
    )

    direction = str(trade.get("direction") or trade.get("side") or "").upper() or "BUY"
    if direction not in {"BUY", "SELL"}:
        direction = "BUY"

    entry_price = _safe_float(trade.get("entry_price"))
    exit_price = _safe_float(trade.get("exit_price"))

    entry_dt = _parse_dt_best_effort(trade.get("entry_time"))
    exit_dt = _parse_dt_best_effort(trade.get("exit_time"))

    entry_time = _to_ist_iso(entry_dt) if entry_dt is not None else _to_ist_iso(trade.get("entry_time"))
    exit_time = _to_ist_iso(exit_dt) if exit_dt is not None else (_to_ist_iso(trade.get("exit_time")) if trade.get("exit_time") else None)

    pnl_val = _safe_float(trade.get("pnl"))
    if pnl_val is None and entry_price is not None and exit_price is not None:
        pnl_points = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
        pnl_val = pnl_points
    pnl_val = float(pnl_val or 0.0)

    pnl_percent = 0.0
    if entry_price is not None and entry_price != 0:
        pnl_percent = float((pnl_val / entry_price) * 100.0)

    exit_reason = trade.get("exit_reason")
    if exit_reason is None:
        exit_reason = trade.get("close_reason")

    timeframe = str(trade.get("timeframe") or trade.get("signal_timeframe") or "")
    strategy = str(trade.get("strategy") or "")
    reason = trade.get("reason")
    if reason is None:
        reason = trade.get("close_reason")

    session = "unknown"
    if entry_dt is not None:
        session = _infer_session_label(entry_dt)

    return {
        "trade_id": str(trade_id),
        "symbol": str(symbol),
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl": pnl_val,
        "pnl_percent": round(pnl_percent, 6),
        "exit_reason": str(exit_reason) if exit_reason is not None else None,
        "timeframe": timeframe,
        "strategy": strategy,
        "reason": str(reason) if reason is not None else None,
        "session": session,
    }


_MCX_EXPIRY_EXIT_DAYS = 2

_LIVE_EXEC_ENABLED = True  # ENABLE live execution for MCX commodities
_LIVE_EXEC_POLL_SECONDS = 2.0
_LIVE_EXEC_MAX_SIGNALS_PER_POLL = 50
_LIVE_EXEC_ONLY_MCX_COMMODITIES = True

_EVENING_LIMIT_BUFFER_POINTS = {
    "GOLD": 1.0,
    "SILVER": 5.0,
    "CRUDEOIL": 1.0,
}

_live_exec_lock = threading.Lock()
_live_exec_processed_signals: Set[str] = set()

# ---------------------------------------------------------------------------
# High-frequency resiliency: global tick health state
# ---------------------------------------------------------------------------

# Raw tick queue — WebSocket puts ticks here, worker thread drains it.
_tick_queue: queue.Queue = queue.Queue(maxsize=20000)

# Last time any tick was received from Fyers (epoch seconds).
_last_tick_time: float = 0.0

# Running counter of ticks received since midnight IST.
_total_ticks_today: int = 0
_ticks_today_date: Optional[date] = None
_tick_count_lock = threading.Lock()

# Background threads (started on app startup, kept alive forever).
_tick_worker_thread: Optional[threading.Thread] = None
_ws_watchdog_running = False

# Reliability & Session tracking
_fyers_token_timestamp: float = time.time()  # Track when the token was last refreshed
_last_crude_tick_time: float = 0.0          # Used for the 3-minute Crude specific watchdog

def _commodity_lot_size(symbol_code: str) -> int:
    sym = str(symbol_code or "").upper()
    if not sym.startswith("MCX:"):
        return 1

    # Multipliers as per user request:
    # Crude: 100, Gold Mini: 10, Silver Mini: 5
    if "CRUDEOIL" in sym:
        return 100
    if "GOLDM" in sym:
        return 10
    if "SILVERM" in sym:
        return 5
    
    # Equity and undefined default to 1
    return 1


def _estimate_brokerage_and_taxes(symbol_code: str, turnover: float) -> float:
    """Calculate estimated brokerage and statutory charges.
    
    Breakdown:
    - Brokerage: 0.02% on turnover (as per user requirement)
    - Other statutory charges (STT, GST, etc.) estimated at 0.02%
    Total: 0.04% on turnover
    """
    brokerage_rate = 0.0002  # 0.02%
    statutory_rate = 0.0002  # 0.02% for STT, GST, etc.
    total_rate = brokerage_rate + statutory_rate  # 0.04% total
    return turnover * total_rate


def _calculate_trade_brokerage(symbol_code: str, entry_price: float, exit_price: float, qty: int = 1) -> Dict[str, float]:
    """Calculate detailed brokerage breakdown for a single trade.
    
    Returns a dict with:
    - entry_turnover: Entry turnover
    - exit_turnover: Exit turnover
    - total_turnover: Total turnover
    - brokerage: Brokerage charges (0.02%)
    - statutory_charges: Other statutory charges (0.02%)
    - total_charges: Total charges
    """
    lot_size = _commodity_lot_size(symbol_code)
    entry_turnover = entry_price * lot_size * qty
    exit_turnover = exit_price * lot_size * qty
    total_turnover = entry_turnover + exit_turnover
    
    brokerage_rate = 0.0002  # 0.02%
    statutory_rate = 0.0002  # 0.02%
    
    brokerage = total_turnover * brokerage_rate
    statutory_charges = total_turnover * statutory_rate
    total_charges = brokerage + statutory_charges
    
    return {
        "entry_turnover": entry_turnover,
        "exit_turnover": exit_turnover,
        "total_turnover": total_turnover,
        "brokerage": brokerage,
        "statutory_charges": statutory_charges,
        "total_charges": total_charges,
    }


def _mcx_days_to_expiry(symbol_code: str) -> Optional[int]:
    sym = str(symbol_code or "").strip()
    if not sym.upper().startswith("MCX:"):
        return None

    _, exp = get_mcx_contract_meta(sym)
    if exp is None:
        return None
    return (exp - date.today()).days


def _mcx_should_block_new_trade(symbol_code: str) -> bool:
    days = _mcx_days_to_expiry(symbol_code)
    if days is None:
        return False
    return days <= _MCX_EXPIRY_EXIT_DAYS


def _is_evening_session_for_mcx() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return now.time() >= dt_time(17, 0)


def _infer_mcx_underlying(symbol_code: str) -> Optional[str]:
    sym = str(symbol_code or "").upper()
    if "GOLD" in sym and "GOLDM" not in sym:
        return "GOLD"
    if "SILVER" in sym and "SILVERM" not in sym and "SILVERMIC" not in sym:
        return "SILVER"
    if "CRUDEOIL" in sym and "CRUDEOILM" not in sym:
        return "CRUDEOIL"
    return None


# Maximum age (seconds) of a cached WS tick before it is considered stale.
# If the last tick for a symbol is older than this, we refuse to trade on it.
_WS_TICK_MAX_AGE_S: float = 60.0


def _is_ws_live() -> bool:
    """Return True only when the Fyers WebSocket has delivered at least one
    tick within the last _WS_TICK_MAX_AGE_S seconds.

    This is the single source of truth used in the Paper Trading engine to
    decide whether it is safe to open or close a position based on live
    market data.  If this returns False, NO trade actions are taken.
    """
    if _last_tick_time <= 0:
        return False
    return (time.time() - _last_tick_time) < _WS_TICK_MAX_AGE_S


def _get_ltp_ws_only(symbol_code: str) -> Optional[float]:
    """Return the latest LTP for *symbol_code* from the WebSocket cache only.

    IMPORTANT: This function deliberately does **NOT** fall back to the
    Fyers REST `quotes` API.  REST calls during a high-frequency trading
    loop would saturate the rate-limit, introduce latency, and create an
    inconsistent data source.  All prices must originate from the live
    WebSocket tick stream.

    Returns None if:
      - The symbol has never received a WS tick this session.
      - The most recent tick is stale (older than _WS_TICK_MAX_AGE_S seconds).
    """
    if not _is_ws_live():
        logger.warning(
            "WS_LTP: WebSocket appears offline or stale (last_tick_age=%.0fs). "
            "Refusing to return price for %s to avoid stale data.",
            time.time() - _last_tick_time if _last_tick_time > 0 else -1,
            symbol_code,
        )
        return None

    ltp = _get_latest_ltp(symbol_code)
    if ltp is None or ltp <= 0:
        logger.info(
            "WS_LTP: No WebSocket tick cached for %s — symbol may not be subscribed.",
            symbol_code,
        )
        return None

    return ltp


def _get_ltp_best_effort(symbol_code: str) -> Optional[float]:
    """DEPRECATED wrapper kept for non-trading display paths only.

    For the paper trading engine use _get_ltp_ws_only() — it enforces
    the 'no REST calls inside the engine' contract strictly.
    The REST fallback below should NEVER be reached from trading code paths.
    """
    ltp = _get_ltp_ws_only(symbol_code)
    if ltp is not None:
        return ltp

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ REST FALLBACK — only for non-critical display paths (e.g.       ║
    # ║ the live order execution panel, NOT the paper engine).           ║
    # ║ This block is intentionally guarded so it is unreachable from   ║
    # ║ _open_paper_trade_from_signal / _update_paper_positions.         ║
    # ╚══════════════════════════════════════════════════════════════════╝
    if _fyers_client is None:
        return None

    try:
        logger.warning(
            "LTP_REST_FALLBACK: fetching price for %s via REST (non-engine path)",
            symbol_code,
        )
        resp = _fyers_client.quotes({"symbols": symbol_code})
    except Exception:
        return None

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        return None

    d = resp.get("d") or []
    if not d:
        return None
    v = d[0].get("v") or {}
    try:
        return float(v.get("lp") or v.get("ltp") or 0.0) or None
    except Exception:
        return None


def _paper_compute_entry_price(
    symbol_code: str,
    direction: str,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """Compute the entry price for a new paper trade using WS LTP exclusively.

    The *fallback* parameter is accepted for interface compatibility but is
    intentionally ignored — we must never open a trade at a price that did
    not come from the live WebSocket feed.
    """
    ltp = _get_ltp_ws_only(symbol_code)
    if ltp is None or ltp <= 0:
        # Hard-fail: no WS data means no trade.
        return None

    direction_up = str(direction or "").upper()
    if not symbol_code.upper().startswith("MCX:"):
        return float(ltp)

    underlying = _infer_mcx_underlying(symbol_code)
    if underlying not in {"GOLD", "SILVER", "CRUDEOIL"}:
        return float(ltp)

    price = float(ltp)
    if _is_evening_session_for_mcx():
        buf = float(_EVENING_LIMIT_BUFFER_POINTS.get(underlying, 0.0))
        if direction_up == "BUY":
            price += buf
        elif direction_up == "SELL":
            price = max(0.0, price - buf)
    return price


def _place_limit_order_live(symbol_code: str, direction: str, qty_lots: int, limit_price: float) -> Optional[Dict[str, Any]]:
    if _fyers_client is None:
        logger.warning("LiveExec: cannot place order; Fyers client not initialized")
        return None

    side = 1 if direction == "BUY" else -1
    order_payload = {
        "symbol": symbol_code,
        "qty": int(qty_lots),
        "type": 2,  # 2 = LIMIT
        "side": side,
        "productType": "MARGIN",
        "limitPrice": float(limit_price),
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": "False",
        "stopLoss": 0,
        "takeProfit": 0,
    }

    method = getattr(_fyers_client, "place_order", None)
    if not callable(method):
        method = getattr(_fyers_client, "placeOrder", None)
    if not callable(method):
        logger.error("LiveExec: Fyers client has no place_order/placeOrder method")
        return None

    try:
        resp = method(order_payload)
    except Exception as e:
        logger.error("LiveExec: order placement failed: %s", e)
        return None

    if not isinstance(resp, dict):
        logger.warning("LiveExec: unexpected place_order response: %r", resp)
        return None

    return resp


async def _live_execution_loop() -> None:
    while True:
        try:
            if not _LIVE_EXEC_ENABLED:
                await asyncio.sleep(_LIVE_EXEC_POLL_SECONDS)
                continue

            if _get_mode() != "live":
                await asyncio.sleep(_LIVE_EXEC_POLL_SECONDS)
                continue

            manager = _engine_manager
            if manager is None:
                await asyncio.sleep(_LIVE_EXEC_POLL_SECONDS)
                continue

            raw_signals = manager.get_signals(limit=200) or []
            signals: List[Dict[str, Any]] = []
            for s in raw_signals:
                if isinstance(s, dict):
                    signals.append(s)
                else:
                    try:
                        signals.append(dict(s))
                    except Exception:
                        continue

            if not signals:
                await asyncio.sleep(_LIVE_EXEC_POLL_SECONDS)
                continue

            processed_this_tick = 0
            for signal in signals:
                if processed_this_tick >= _LIVE_EXEC_MAX_SIGNALS_PER_POLL:
                    break

                sid = _make_signal_id(signal)
                with _live_exec_lock:
                    if sid in _live_exec_processed_signals:
                        continue

                symbol_raw = str(signal.get("symbol") or "")
                symbol_code = _normalize_instrument(symbol_raw)
                if not symbol_code:
                    continue

                sym_up = symbol_code.upper()
                if _LIVE_EXEC_ONLY_MCX_COMMODITIES and not sym_up.startswith("MCX:"):
                    continue

                underlying = _infer_mcx_underlying(symbol_code)
                if underlying not in {"GOLD", "SILVER", "CRUDEOIL"}:
                    continue

                if not _is_market_open_for_instrument(symbol_code):
                    continue

                if _mcx_should_block_new_trade(symbol_code):
                    days = _mcx_days_to_expiry(symbol_code)
                    logger.info(
                        "LiveExec: blocked new trade for %s near expiry (days_to_expiry=%s)",
                        symbol_code,
                        days,
                    )
                    continue

                direction = str(signal.get("final_action") or signal.get("action") or "").upper()
                if direction not in {"BUY", "SELL"}:
                    continue

                qty_lots = 1
                ltp = _get_ltp_ws_only(symbol_code)
                if ltp is None or ltp <= 0:
                    logger.info(
                        "LiveExec: skip %s – no WS LTP available (symbol may not be subscribed)",
                        symbol_code,
                    )
                    continue

                # Bypass SMC validation for simple EMA scalping strategy
                contributing = signal.get("contributing_strategies") or []
                if "ema_scalping_5" in contributing:
                    smc_valid = True
                    smc_delay = False
                else:
                    smc = is_smc_valid(symbol_code, direction, float(ltp), ltf_timeframe="5m", htf_timeframe="60m")
                    smc_valid = smc.valid
                    smc_delay = smc.delay

                if not smc_valid:
                    if smc_delay:
                        continue
                    with _live_exec_lock:
                        _live_exec_processed_signals.add(sid)
                    continue

                limit_price = float(ltp)
                if _is_evening_session_for_mcx():
                    buf = float(_EVENING_LIMIT_BUFFER_POINTS.get(underlying, 0.0))
                    if direction == "BUY":
                        limit_price = limit_price + buf
                    else:
                        limit_price = max(0.0, limit_price - buf)

                if smc.order_type == "LIMIT" and smc.limit_price is not None:
                    limit_price = float(smc.limit_price)

                resp = _place_limit_order_live(symbol_code, direction, qty_lots, limit_price)
                logger.info(
                    "LiveExec: placed %s LIMIT %s qty=%s price=%.4f resp=%r",
                    direction,
                    symbol_code,
                    qty_lots,
                    limit_price,
                    resp,
                )

                processed_this_tick += 1

                with _live_exec_lock:
                    _live_exec_processed_signals.add(sid)

        except Exception as e:
            logger.error("LiveExec loop error: %s", e)

        await asyncio.sleep(_LIVE_EXEC_POLL_SECONDS)


def _to_ist_iso(ts: Any) -> str:
    """Formats timestamp to standard IST human-readable format: DD-MMM-YYYY hh:mm AM/PM"""
    if ts is None or ts == "":
        return ""
    try:
        dt = pd.to_datetime(ts)
    except Exception:
        return str(ts)

    try:
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.tz_localize(IST)  # type: ignore[union-attr]
        else:
            dt = dt.tz_convert(IST)  # type: ignore[union-attr]
        py = dt.to_pydatetime()  # type: ignore[union-attr]
        return py.strftime("%d-%b-%Y %I:%M %p")
    except Exception:
        try:
            py2 = dt.to_pydatetime()  # type: ignore[union-attr]
            if getattr(py2, "tzinfo", None) is None:
                py2 = IST.localize(py2)
            else:
                py2 = py2.astimezone(IST)
            return py2.strftime("%d-%b-%Y %I:%M %p")
        except Exception:
            return str(ts)


def _load_stored_signals_history(limit: int = 2000) -> List[Dict[str, Any]]:
    lim = max(1, min(int(limit), 20000))
    base_dir = os.path.join(BASE_DIR, "signals")
    if not os.path.isdir(base_dir):
        return []

    collected: List[Dict[str, Any]] = []

    # Hard cutoff: only show signals generated (wall-clock) within 7 days
    cutoff_dt = datetime.now(IST) - timedelta(days=7)

    # Scan last 7 days of signal directories
    for days_back in range(8):
        d = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        fp = os.path.join(base_dir, d, f"signals_{d}.json")
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    # Filter by timestamp_generated (when the signal was ACTUALLY created)
                    ts_gen = item.get("timestamp_generated") or item.get("signal_timestamp") or ""
                    if ts_gen:
                        try:
                            # Try custom IST format first
                            if "AM" in str(ts_gen) or "PM" in str(ts_gen):
                                parsed = datetime.strptime(str(ts_gen), "%d-%b-%Y %I:%M %p")
                                parsed = IST.localize(parsed)
                            else:
                                parsed = datetime.fromisoformat(str(ts_gen))
                                if parsed.tzinfo is None:
                                    parsed = IST.localize(parsed)
                                else:
                                    parsed = parsed.astimezone(IST)
                            
                            if parsed < cutoff_dt:
                                continue  # Skip stale signals from before the cutoff
                        except Exception:
                            # If unparseable, we assume it's old if we're scanning multiple days back
                            if days_back > 1:
                                continue
                    collected.append(item)
        except Exception:
            continue

        if len(collected) >= lim:
            break

    # Sort by timestamp_generated (actual wall-clock creation time), newest first
    def _ts_gen_key(s: Dict[str, Any]) -> str:
        return str(s.get("timestamp_generated") or s.get("signal_timestamp") or s.get("timestamp") or "")

    collected.sort(key=_ts_gen_key, reverse=True)
    return collected[:lim]


def _merge_and_dedupe_signals(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for s in primary + secondary:
        if not isinstance(s, dict):
            continue
        try:
            sid = _make_signal_id(s)
        except Exception:
            sid = str(uuid.uuid4())
        if sid in seen:
            continue
        seen.add(sid)
        out.append(s)
    return out


def _paper_trades_storage_dir() -> str:
    return os.path.join(BASE_DIR, "trades")


def _paper_trades_file_for_date(d: date) -> str:
    ds = d.strftime("%Y-%m-%d")
    return os.path.join(_paper_trades_storage_dir(), ds, f"paper_trades_{ds}.json")


def _persist_paper_trades_snapshot(trades: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(_paper_trades_storage_dir(), exist_ok=True)
        fp = _paper_trades_file_for_date(date.today())
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception as e:
        logger.error("persist_paper_trades_snapshot error: %s", e)


def _load_stored_paper_trades_history(limit: int = 5000) -> List[Dict[str, Any]]:
    lim = max(1, min(int(limit), 20000))
    base_dir = _paper_trades_storage_dir()
    if not os.path.isdir(base_dir):
        return []

    collected: List[Dict[str, Any]] = []
    for days_back in range(30):
        d = (date.today() - timedelta(days=days_back))
        fp = _paper_trades_file_for_date(d)
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Clean up corrupted data: remove duplicate uppercase keys
                        cleaned = _cleanup_trade_dict(item)
                        collected.append(cleaned)
        except Exception:
            continue

        if len(collected) >= lim:
            break

    def _ts_key(t: Dict[str, Any]) -> str:
        return str(t.get("exit_time") or t.get("entry_time") or "")

    collected.sort(key=_ts_key, reverse=True)
    return collected[:lim]


def _cleanup_trade_dict(trade: Dict[str, Any]) -> Dict[str, Any]:
    """Remove duplicate uppercase keys that were accidentally added to stored trades.
    
    Keeps only the original lowercase keys (symbol, direction, etc.) and removes
    the uppercase duplicates (Symbol, Direction, etc.) that were added by the UI.
    """
    # Keys that were duplicated (lowercase is original, uppercase is duplicate)
    duplicate_keys = {
        'symbol': 'Symbol',
        'direction': 'Direction', 
        'timeframe': 'Timeframe',
        'reason': 'Reason',
        'outcome': 'Outcome',
    }
    
    cleaned = dict(trade)
    
    # Remove uppercase duplicates, keeping lowercase originals
    for lower_key, upper_key in duplicate_keys.items():
        if upper_key in cleaned and lower_key in cleaned:
            # Both exist - remove the uppercase one (it's the duplicate)
            del cleaned[upper_key]
        elif upper_key in cleaned and lower_key not in cleaned:
            # Only uppercase exists - this shouldn't happen, but keep it as lowercase
            cleaned[lower_key] = cleaned.pop(upper_key)
    
    # Also remove other UI-added fields that shouldn't be persisted
    ui_only_fields = [
        'display_symbol', 'id', 'Sr. No', 'Entry Price', 'Exit Price',
        'Entry Time (IST)', 'Exit Time (IST)', 'Points Captured',
        'P&L Excl. Brokerage', 'Brokerage', 'Brokerage Breakdown'
    ]
    
    for field in ui_only_fields:
        cleaned.pop(field, None)
    
    return cleaned


def _make_trade_id(trade: Dict[str, Any]) -> str:
    tid = trade.get("trade_id") or trade.get("id")
    if tid is None:
        tid = ""
    tid = str(tid)
    if tid:
        return tid

    symbol = str(trade.get("symbol") or trade.get("instrument") or "")
    direction = str(trade.get("direction") or trade.get("side") or "")
    entry_time = str(trade.get("entry_time") or "")
    entry_price = str(trade.get("entry_price") or "")
    return f"{symbol}|{direction}|{entry_time}|{entry_price}"


def _merge_and_dedupe_trades(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for t in primary + secondary:
        if not isinstance(t, dict):
            continue
        tid = _make_trade_id(t)
        if tid in seen:
            continue
        seen.add(tid)
        out.append(t)
    return out

# Full Nifty-50 universe used for the dashboard watchlist. Keys are display
# symbols (like "RELIANCE"), values are Fyers instrument codes.
NIFTY_50_SYMBOLS: Dict[str, str] = {
    "RELIANCE": "NSE:RELIANCE-EQ",
    "TCS": "NSE:TCS-EQ",
    "INFY": "NSE:INFY-EQ",
    "HDFCBANK": "NSE:HDFCBANK-EQ",
    "ICICIBANK": "NSE:ICICIBANK-EQ",
    "KOTAKBANK": "NSE:KOTAKBANK-EQ",
    "LT": "NSE:LT-EQ",
    "ITC": "NSE:ITC-EQ",
    "SBIN": "NSE:SBIN-EQ",
    "HINDUNILVR": "NSE:HINDUNILVR-EQ",
    "AXISBANK": "NSE:AXISBANK-EQ",
    "BAJFINANCE": "NSE:BAJFINANCE-EQ",
    "ASIANPAINT": "NSE:ASIANPAINT-EQ",
    "MARUTI": "NSE:MARUTI-EQ",
    "SUNPHARMA": "NSE:SUNPHARMA-EQ",
    "TITAN": "NSE:TITAN-EQ",
    "WIPRO": "NSE:WIPRO-EQ",
    "ULTRACEMCO": "NSE:ULTRACEMCO-EQ",
    "NESTLEIND": "NSE:NESTLEIND-EQ",
    "POWERGRID": "NSE:POWERGRID-EQ",
    "BAJAJFINSV": "NSE:BAJAJFINSV-EQ",
    "TECHM": "NSE:TECHM-EQ",
    "NTPC": "NSE:NTPC-EQ",
    "GRASIM": "NSE:GRASIM-EQ",
    "JSWSTEEL": "NSE:JSWSTEEL-EQ",
    "HCLTECH": "NSE:HCLTECH-EQ",
    "TATAMOTORS": "NSE:TATAMOTORS-EQ",
    "DRREDDY": "NSE:DRREDDY-EQ",
    "CIPLA": "NSE:CIPLA-EQ",
    "ONGC": "NSE:ONGC-EQ",
    "HDFCLIFE": "NSE:HDFCLIFE-EQ",
    "DIVISLAB": "NSE:DIVISLAB-EQ",
    "HEROMOTOCO": "NSE:HEROMOTOCO-EQ",
    "BRITANNIA": "NSE:BRITANNIA-EQ",
    "BPCL": "NSE:BPCL-EQ",
    "COALINDIA": "NSE:COALINDIA-EQ",
    "ADANIENT": "NSE:ADANIENT-EQ",
    "ADANIPORTS": "NSE:ADANIPORTS-EQ",
    "INDUSINDBK": "NSE:INDUSINDBK-EQ",
    "BAJAJ-AUTO": "NSE:BAJAJ-AUTO-EQ",
    "EICHERMOT": "NSE:EICHERMOT-EQ",
    "TATACONSUM": "NSE:TATACONSUM-EQ",
    "HINDALCO": "NSE:HINDALCO-EQ",
    "APOLLOHOSP": "NSE:APOLLOHOSP-EQ",
    "TATASTEEL": "NSE:TATASTEEL-EQ",
    "M&M": "NSE:M&M-EQ",
    "BHARTIARTL": "NSE:BHARTIARTL-EQ",
    "SHRIRAMFIN": "NSE:SHRIRAMFIN-EQ",
    "JIOFINANCE": "NSE:JIOFIN-EQ",
    "UPL": "NSE:UPL-EQ",
}

COMMODITY_SYMBOLS: Dict[str, str] = {
    "GOLD": "MCX:GOLD26APRFUT",
    "SILVER": "MCX:SILVER26MARFUT", 
    "CRUDE": "MCX:CRUDEOIL26FEBFUT",
}

_ALL_SYMBOLS_MAP: Dict[str, str] = {}
INSTRUMENT_TO_DISPLAY: Dict[str, str] = {}


def _rebuild_instrument_maps() -> None:
    _ALL_SYMBOLS_MAP.clear()
    _ALL_SYMBOLS_MAP.update(NIFTY_50_SYMBOLS)
    _ALL_SYMBOLS_MAP.update(COMMODITY_SYMBOLS)

    INSTRUMENT_TO_DISPLAY.clear()
    for display_symbol, instrument_code in _ALL_SYMBOLS_MAP.items():
        INSTRUMENT_TO_DISPLAY[instrument_code] = display_symbol


_rebuild_instrument_maps()

def _init_commodity_symbols() -> None:
    """Initialize commodity symbols with current MCX contract codes."""
    
    # Use current MCX contract codes for real-time data
    mapping = {
        "GOLD": get_current_commodity_symbol("GOLD") or "MCX:GOLD26APRFUT",
        "SILVER": get_current_commodity_symbol("SILVER") or "MCX:SILVER26MARFUT",
        "CRUDE": get_current_commodity_symbol("CRUDEOIL") or "MCX:CRUDEOIL26FEBFUT",
    }
    
    if not mapping:
        logger.warning("Could not resolve any MCX commodity symbols; using empty commodities watchlist")
        return
    
    COMMODITY_SYMBOLS.clear()
    COMMODITY_SYMBOLS.update(mapping)
    _rebuild_instrument_maps()
    logger.info("Updated commodity symbols with current MCX contracts: %s", list(mapping.keys()))


# Trading engine manager (runs LiveEngine in background thread)
_engine_manager: Optional[EngineManager] = None
_engine_lock = threading.Lock()

# Fyers REST client for account / quotes / positions
_fyers_client: Optional[fyersModel.FyersModel] = None
_fyers_lock = threading.Lock()

# Dashboard mode: "live" uses real Fyers + engine, "paper" uses paper trading
_APP_MODE: str = os.getenv("MINI_SIMON_MODE", "live").lower()
_mode_lock = threading.Lock()

# In-memory caches for high-frequency UI access
_market_feed_cache: Dict[str, Dict[str, Any]] = {}
_commodity_feed_cache: Dict[str, Dict[str, Any]] = {}
_positions_cache: List[Dict[str, Any]] = []
_orders_cache: List[Dict[str, Any]] = []
_margin_cache: Dict[str, Any] = {}
_market_feed_lock = threading.Lock()
_commodity_feed_lock = threading.Lock()

# Optional WebSocket manager for live ticks powering the watchlist.
_ws_manager: Optional["DashboardWebSocketManager"] = None
_ws_lock = threading.Lock()

# Live candle manager for the price chart, updated from WebSocket ticks.
_candle_manager: Optional[LiveCandleManager] = None
_candle_lock = threading.Lock()

_paper_lock = threading.Lock()

# Clear any existing fake trades and positions
_paper_trades: List[Dict[str, Any]] = []
_paper_positions: Dict[str, Dict[str, Any]] = {}
_paper_trades_version: int = 0
_paper_processed_signals: Set[str] = set()
_paper_engine_running: bool = False
_paper_last_heartbeat: float = 0.0

_backtest_jobs: Dict[str, Dict[str, Any]] = {}
_backtest_jobs_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Browser WebSocket for market feed snapshots
# -----------------------------------------------------------------------------


async def _get_market_rows_snapshot() -> List[Dict[str, Any]]:
    """Return a sorted snapshot of the current market feed cache."""

    with _market_feed_lock:
        rows = sorted(_market_feed_cache.values(), key=lambda r: r.get("symbol") or "")
    return rows


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _is_market_open_india() -> bool:
    """Rough check for NSE cash session (09:15"+-15:30 IST, Mon-Fri).

    This is intentionally simple. For precise status we can also consult
    fyers.market_status() when needed.
    """

    now = datetime.now(IST)
    # Monday=0, Sunday=6
    if now.weekday() >= 5:
        return False
    return dt_time(9, 15) <= now.time() <= dt_time(15, 30)


def _is_equity_market_open() -> bool:
    return _is_market_open_india()


def _is_commodity_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return dt_time(9, 0) <= now.time() <= dt_time(23, 30)


def _is_market_open_for_instrument(symbol_code: str) -> bool:
    """Check if market is open based on instrument type and current IST time.
    9:15 AM – 3:30 PM: Equity.
    9:00 AM - 11:30 PM: MCX.
    """
    now = datetime.now(IST)
    if now.weekday() >= 5: # Weekend
        return False
        
    current_time = now.time()
    is_mcx = symbol_code.upper().startswith("MCX:")
    
    if is_mcx:
        return dt_time(9, 0) <= current_time <= dt_time(23, 30)
    else:
        return dt_time(9, 15) <= current_time <= dt_time(15, 30)


def _get_mode() -> str:
    with _mode_lock:
        return _APP_MODE


def _set_mode(mode: str) -> None:
    """Set dashboard mode.

    Externally we expose only two modes:
    - "live"   : show broker account data (positions/orders/margin) + live quotes
    - "paper" : paper trading view (no real orders are placed by this UI),
                still using real market data when available.

    For backward compatibility we accept historical values like
    "sim" / "simulated" / "paper_trading" but normalise them to "paper".
    """

    global _APP_MODE
    raw = (mode or "").lower()

    # Backwards-compatible aliases
    if raw in {"sim", "simulated", "paper_trading"}:
        mode_normalised = "paper"
    elif raw in {"live", "paper"}:
        mode_normalised = raw
    else:
        raise ValueError("Mode must be 'live' or 'paper'")

    with _mode_lock:
        _APP_MODE = mode_normalised


def _init_logging() -> None:
    """Initialize logging using existing LoggerConfig if available."""

    try:
        LoggerConfig.setup_logging()
    except Exception:
        # Fallback basic config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _init_engine() -> None:
    """Start LiveEngine via EngineManager in a background thread.

    Core trading logic remains untouched; we only control its lifecycle here.
    """

    global _engine_manager
    with _engine_lock:
        if _engine_manager is not None:
            return

        _engine_manager = EngineManager()

        data_cfg = _engine_manager.config.get("data_feed", {})

        # Wire Fyers credentials from the same sources used by the dashboard
        # so the LiveEngine can connect its own data feed (WS/REST).
        auth = _get_fyers_ws_auth()
        if auth is not None:
            app_id, websocket_token = auth
            data_cfg["app_id"] = app_id
            # For Fyers v3, both REST `token` and WS `access_token` expect
            # the "APP_ID:ACCESS_TOKEN" format, which `_get_fyers_ws_auth`
            # already returns as `websocket_token`.
            data_cfg["access_token"] = websocket_token

        symbols: List[str] = list(NIFTY_50_SYMBOLS.values())
        if COMMODITY_SYMBOLS:
            symbols += list(COMMODITY_SYMBOLS.values())
        data_cfg["symbols"] = symbols
        data_cfg["timeframes"] = [
            "1m",
            "3m",
            "5m",
            "15m",
            "60m",
            "1D",
        ]

        _engine_manager.config["data_feed"] = data_cfg

        # In paper mode, relax signal aggregation thresholds slightly so that
        # some candidate signals make it through for paper-trade testing.
        agg_cfg = _engine_manager.config.get("signal_aggregator", {})
        if _APP_MODE == "paper":
            try:
                # Lower the minimum confidence and require only a single
                # strategy confluence to emit a signal.
                agg_cfg["min_confidence_threshold"] = float(
                    agg_cfg.get("min_confidence_threshold", 0.3) * 0.5
                )
                agg_cfg["confluence_threshold"] = 1
            except Exception:
                # Fallback to safe, explicit values if anything is weird.
                agg_cfg["min_confidence_threshold"] = 0.15
                agg_cfg["confluence_threshold"] = 1

        _engine_manager.config["signal_aggregator"] = agg_cfg

        started = _engine_manager.start_engine()
        if not started:
            logger.warning("LiveEngine failed to start from FastAPI backend")
        else:
            logger.info("LiveEngine started in background thread")


def _init_fyers_client() -> None:
    """Initialize Fyers REST client using config.data_feed credentials."""

    global _fyers_client
    if _fyers_client is not None:
        return

    cfg = get_config()
    app_id = cfg.get("data_feed.app_id")
    access_token = cfg.get("data_feed.access_token")

    # Fallback 1: read raw access token from access.txt if not in config/env.
    if access_token is None or access_token == "YOUR_ACCESS_TOKEN":
        token_path = os.path.join(BASE_DIR, "access.txt")
        if os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    access_token = f.read().strip()
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error reading access.txt for Fyers token: {e}")

    # Fallback 2: derive app_id from a pre-prefixed token like APPID:XXXX.
    if app_id is None or app_id == "YOUR_APP_ID":
        if access_token and ":" in str(access_token):
            app_id = str(access_token).split(":", 1)[0]

    # If token is provided in APP_ID:ACCESS_TOKEN format, use only the raw
    # access token for REST authentication.
    rest_token = str(access_token or "").strip()
    if ":" in rest_token:
        prefix, raw = rest_token.split(":", 1)
        if (app_id is None or app_id == "YOUR_APP_ID") and prefix:
            app_id = prefix
        rest_token = raw

    # Fallback 3: parse client_id from credentials.py if still unknown.
    if app_id is None or app_id == "YOUR_APP_ID":
        cred_path = os.path.join(BASE_DIR, "credentials.py")
        if os.path.exists(cred_path):
            try:
                with open(cred_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_stripped = line.strip()
                        if line_stripped.startswith("client_id") and "=" in line_stripped:
                            _, rhs = line_stripped.split("=", 1)
                            rhs = rhs.strip()
                            if (rhs.startswith('"') and rhs.endswith('"')) or (
                                rhs.startswith("'") and rhs.endswith("'")
                            ):
                                app_id = rhs[1:-1]
                            break
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error parsing credentials.py for client_id: {e}")

    if not app_id or app_id == "YOUR_APP_ID" or not rest_token or rest_token == "YOUR_ACCESS_TOKEN":
        logger.warning("Fyers credentials not configured; live market data will be disabled")
        _fyers_client = None
        return

    try:
        _fyers_client = fyersModel.FyersModel(
            client_id=app_id,
            token=rest_token,
            log_path="",
        )
        logger.info("Fyers REST client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Fyers client: {e}")
        _fyers_client = None


def _get_fyers_ws_auth() -> Optional[Tuple[str, str]]:
    """Return (app_id, websocket_token) for Fyers v3 WebSocket.

    The websocket_token is formatted as required by FyersDataSocket:
    "APP_ID:ACCESS_TOKEN". We reuse the same credential sources and
    fallbacks as _init_fyers_client.
    """

    cfg = get_config()
    app_id = cfg.get("data_feed.app_id")
    access_token_value = cfg.get("data_feed.access_token")

    if app_id == "YOUR_APP_ID":
        app_id = None
    if access_token_value == "YOUR_ACCESS_TOKEN":
        access_token_value = None

    # Fallback 1: read raw access token from access.txt if not in config/env.
    if access_token_value is None:
        token_path = os.path.join(BASE_DIR, "access.txt")
        if os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    access_token_value = f.read().strip()
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error reading access.txt for Fyers token (WS): {e}")

    # Fallback 2: derive app_id from a pre-prefixed token like APPID:XXXX.
    if app_id is None and access_token_value is not None and ":" in str(access_token_value):
        app_id = str(access_token_value).split(":", 1)[0]

    # Fallback 3: parse client_id from credentials.py if still unknown.
    if app_id is None:
        cred_path = os.path.join(BASE_DIR, "credentials.py")
        if os.path.exists(cred_path):
            try:
                with open(cred_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_stripped = line.strip()
                        if line_stripped.startswith("client_id") and "=" in line_stripped:
                            _, rhs = line_stripped.split("=", 1)
                            rhs = rhs.strip()
                            if (rhs.startswith("\"") and rhs.endswith("\"")) or (
                                rhs.startswith("'") and rhs.endswith("'")
                            ):
                                app_id = rhs[1:-1]
                            break
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error parsing credentials.py for client_id (WS): {e}")

    if not app_id or not access_token_value:
        logger.warning("Fyers WebSocket credentials not configured; live tick feed will be disabled")
        return None

    raw_token = str(access_token_value).strip()
    if ":" in raw_token:
        prefix, token_only = raw_token.split(":", 1)
        if not app_id and prefix:
            app_id = prefix
        raw_token = token_only

    websocket_token = f"{app_id}:{raw_token}"

    return app_id, websocket_token


def _get_watchlist_symbols() -> List[str]:
    """Return symbols to monitor in the live market feed.

    Uses the full Nifty-50 universe defined in NIFTY_50_SYMBOLS.
    """

    return list(NIFTY_50_SYMBOLS.keys())


def _get_commodity_symbols() -> List[str]:
    """Return display symbols to monitor in the commodities watchlist."""

    return list(COMMODITY_SYMBOLS.keys())


def _refresh_market_feed_live() -> None:
    """Refresh live market feed cache from Fyers quotes API.

    Designed to be reasonably light so it can be polled every few seconds
    from the frontend during market hours.
    """

    if _fyers_client is None:
        return

    display_symbols = _get_watchlist_symbols()
    if not display_symbols:
        return

    # Map display symbols like "RELIANCE" to full Fyers instrument codes like
    # "NSE:RELIANCE-EQ" using the NIFTY_50_SYMBOLS mapping. This ensures we
    # always call the quotes API with the exact EXCHANGE:SYMBOL-EQ format
    # required by Fyers v3.
    instrument_codes: List[str] = []
    for sym in display_symbols:
        code = NIFTY_50_SYMBOLS.get(sym)
        if code:
            instrument_codes.append(code)

    if not instrument_codes:
        return

    # Fyers quotes API has a limit on how many symbols can be requested in a
    # single call. To stay within limits for the full Nifty-50 basket we batch
    # the symbols and merge the results into one cache.
    max_batch = 20
    now = datetime.now(IST).isoformat()

    # Clear and repopulate cache while holding a short-lived lock so
    # concurrent WebSocket updates remain thread-safe.
    with _market_feed_lock:
        _market_feed_cache.clear()

    for i in range(0, len(instrument_codes), max_batch):
        batch_codes = instrument_codes[i : i + max_batch]
        symbols_param = ",".join(batch_codes)

        try:
            resp = _fyers_client.quotes({"symbols": symbols_param})
        except Exception as e:
            logger.error(f"Error fetching quotes from Fyers for batch {batch_codes}: {e}")
            continue

        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"Unexpected quotes response for batch {batch_codes}: {resp}")
            continue

        data_list = resp.get("d") or []
        for item in data_list:
            name = item.get("n", "")  # e.g. "NSE:RELIANCE" or "NSE:RELIANCE-EQ"
            value = item.get("v") or {}
            if not isinstance(value, dict):
                continue

            # Derive a compact symbol name
            symbol = name.replace("NSE:", "").replace("MCX:", "").replace("-EQ", "") if name else "UNKNOWN"

            # Fyers uses various keys like lp/ltp, chp, etc; we only rely on safe .get()
            ltp = value.get("lp") or value.get("ltp") or 0.0
            volume = value.get("volume") or value.get("tv") or 0.0
            change_pct = value.get("chp") or value.get("change_percent") or 0.0

            with _market_feed_lock:
                _market_feed_cache[symbol] = {
                    "symbol": symbol,
                    "raw_symbol": name,
                    "ltp": float(ltp) if ltp is not None else 0.0,
                    "volume": float(volume) if volume is not None else 0.0,
                    "change_pct": float(change_pct) if change_pct is not None else 0.0,
                    "updated_at": now,
                    "raw": value,
                }

def _refresh_commodity_feed_live() -> None:
    """Refresh commodities feed cache from Fyers quotes API.

    This mirrors _refresh_market_feed_live but uses COMMODITY_SYMBOLS and a
    separate in-memory cache for commodities watchlist panel.
    """
    
    if _fyers_client is None:
        logger.warning("Fyers client unavailable - cannot fetch commodity data")
        return
    
    display_symbols = _get_commodity_symbols()
    if not display_symbols:
        return

    instrument_codes: List[str] = []
    for sym in display_symbols:
        code = COMMODITY_SYMBOLS.get(sym)
        if code:
            instrument_codes.append(code)

    if not instrument_codes:
        return

    max_batch = 20
    now = datetime.now(IST).isoformat()

    with _commodity_feed_lock:
        _commodity_feed_cache.clear()

    for i in range(0, len(instrument_codes), max_batch):
        batch_codes = instrument_codes[i : i + max_batch]
        symbols_param = ",".join(batch_codes)

        try:
            resp = _fyers_client.quotes({"symbols": symbols_param})
        except Exception as e:
            logger.error(f"Error fetching commodity quotes from Fyers for batch {batch_codes}: {e}")
            continue

        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"Unexpected quotes response for commodities batch {batch_codes}: {resp}")
            continue

        data_list = resp.get("d") or []
        for item in data_list:
            name = item.get("n", "")
            value = item.get("v") or {}
            if not isinstance(value, dict):
                continue

            # Use the same instrument-to-display mapping as the main
            # watchlist so symbols like "MCX:GOLD-INDEX" render as "GOLD".
            display_symbol = INSTRUMENT_TO_DISPLAY.get(
                name,
                name.replace("NSE:", "").replace("MCX:", "").replace("-EQ", "") if name else "UNKNOWN",
            )

            ltp = value.get("lp") or value.get("ltp") or 0.0
            volume = value.get("volume") or value.get("tv") or 0.0
            change_pct = value.get("chp") or value.get("change_percent") or 0.0

            with _commodity_feed_lock:
                _commodity_feed_cache[display_symbol] = {
                    "symbol": display_symbol,
                    "raw_symbol": name,
                    "ltp": float(ltp) if ltp is not None else 0.0,
                    "volume": float(volume) if volume is not None else 0.0,
                    "change_pct": float(change_pct) if change_pct is not None else 0.0,
                    "updated_at": now,
                    "raw": value,
                }


def _refresh_positions_live() -> None:
    """Refresh positions cache from Fyers positions() API.

    The exact response structure may vary; we try to normalize a few common
    fields but also keep the raw dict so the UI can inspect everything.
    """

    if _fyers_client is None:
        return

    try:
        resp = _fyers_client.positions()
    except Exception as e:
        logger.error(f"Error fetching positions from Fyers: {e}")
        return

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        logger.warning(f"Unexpected positions response: {resp}")
        return

    items = (
        resp.get("netPositions")
        or resp.get("positions")
        or resp.get("data")
        or []
    )

    _positions_cache.clear()
    for pos in items:
        if not isinstance(pos, dict):
            continue
        symbol = pos.get("symbol") or pos.get("symbolName") or pos.get("scrip")
        qty = pos.get("netQty") or pos.get("qty") or pos.get("quantity")
        buy_val = pos.get("buyVal") or pos.get("buyValue")
        sell_val = pos.get("sellVal") or pos.get("sellValue")
        pnl = pos.get("realized_profit") or pos.get("pnl") or pos.get("unrealized")

        _positions_cache.append(
            {
                "symbol": symbol,
                "qty": qty,
                "buy_value": buy_val,
                "sell_value": sell_val,
                "pnl": pnl,
                "raw": pos,
            }
        )


def _refresh_orders_live() -> None:
    """Refresh recent orders cache from Fyers orderbook() API."""

    if _fyers_client is None:
        return

    try:
        resp = _fyers_client.orderbook()
    except Exception as e:
        logger.error(f"Error fetching orderbook from Fyers: {e}")
        return

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        logger.warning(f"Unexpected orderbook response: {resp}")
        return

    items = resp.get("orderBook") or resp.get("orders") or resp.get("data") or []

    _orders_cache.clear()
    for order in items:
        if not isinstance(order, dict):
            continue
        symbol = order.get("symbol") or order.get("symbolName")
        side = order.get("side") or order.get("transaction_type")
        qty = order.get("qty") or order.get("quantity")
        price = order.get("limitPrice") or order.get("price")
        status = order.get("status") or order.get("orderStatus")
        ts = order.get("orderDateTime") or order.get("timestamp")

        _orders_cache.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "status": status,
                "timestamp": ts,
                "raw": order,
            }
        )


def _refresh_margin_live() -> None:
    """Refresh margin / funds cache from Fyers funds() API."""

    if _fyers_client is None:
        return

    try:
        resp = _fyers_client.funds()
    except Exception as e:
        logger.error(f"Error fetching funds from Fyers: {e}")
        return

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        logger.warning(f"Unexpected funds response: {resp}")
        return

    # The exact structure depends on Fyers; we keep it mostly as-is.
    _margin_cache.clear()
    for k, v in resp.items():
        if k == "s":
            continue
        _margin_cache[k] = v


def _refresh_all_live() -> None:
    """Refresh all live caches in a single call."""

    _refresh_positions_live()
    _refresh_orders_live()
    _refresh_margin_live()


BACKTEST_RESOLUTION_MAP: Dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1d": "D",
    "60m": "60",
    "120m": "120",
    "240m": "240",
    "1d": "D",
    "1D": "D",
}


def _normalize_backtest_timeframe(tf: str) -> str:
    t = str(tf or "").strip()
    if not t:
        return t

    tl = t.lower()
    if tl in {"d", "1d", "day", "1day"}:
        return "1D"

    if tl.endswith("h"):
        try:
            hours = int(tl[:-1])
            return f"{hours * 60}m"
        except Exception:
            return t

    if tl.endswith("m"):
        try:
            mins = int(tl[:-1])
            return f"{mins}m"
        except Exception:
            return t

    return t


def _to_yyyy_mm_dd(v: str) -> str:
    s = str(v or "").strip()
    if not s:
        return ""

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d")
    except Exception:
        return s


def _backtest_timeframe_to_fyers_resolution(tf: str) -> str:
    tf_norm = _normalize_backtest_timeframe(tf)
    tf_key = str(tf_norm or "").strip()
    return BACKTEST_RESOLUTION_MAP.get(tf_key, BACKTEST_RESOLUTION_MAP.get(tf_key.lower(), tf_key))


def _resolve_backtest_instrument(segment: str, display_symbol: str) -> str:
    seg = str(segment or "").strip().upper()
    sym = str(display_symbol or "").strip().upper()

    if seg == "MCX":
        code = COMMODITY_SYMBOLS.get(sym)
        if code:
            return code
        if sym.startswith("MCX:"):
            return sym
        return f"MCX:{sym}"

    if sym.startswith("NSE:") and sym.endswith("-EQ"):
        return sym
    if ":" in sym:
        return sym
    return f"NSE:{sym}-EQ"


def _fetch_history_df(
    symbol_code: str,
    timeframe: str,
    date_from: str,
    date_to: str,
    display_symbol: str = "",
) -> Tuple[pd.DataFrame, Any]:
    if _fyers_client is None:
        return pd.DataFrame(), None

    resolution = _backtest_timeframe_to_fyers_resolution(timeframe)
    range_from = _to_yyyy_mm_dd(date_from)
    range_to = _to_yyyy_mm_dd(date_to)
    payload = {
        "symbol": symbol_code,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1",
    }
    try:
        resp = _fyers_client.history(payload)
    except Exception as e:
        print(f"FAILED: {display_symbol or symbol_code} at {timeframe}. Exception: {e}")
        return pd.DataFrame(), {"exception": str(e)}

    if resp is None or (not isinstance(resp, dict)) or ("candles" not in resp) or (not resp.get("candles")):
        print(f"FAILED: {display_symbol or symbol_code} at {timeframe}. Response: {resp}")
        return pd.DataFrame(), resp

    candles = resp.get("candles") or []
    if len(candles) == 0:
        print(f"FAILED: {display_symbol or symbol_code} at {timeframe}. Response: {resp}")
        return pd.DataFrame(), resp

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
    return df[["date", "open", "high", "low", "close", "volume"]], resp


def _resolve_mcx_active_contracts() -> None:
    """Refresh COMMODITY_SYMBOLS for key commodities using the MCX symbol master."""

    try:
        updated = False
        gold_sym = get_current_commodity_symbol("GOLD")
        if gold_sym:
            COMMODITY_SYMBOLS["GOLD"] = gold_sym
            updated = True

        silver_sym = get_current_commodity_symbol("SILVER")
        if silver_sym:
            COMMODITY_SYMBOLS["SILVER"] = silver_sym
            updated = True

        crude_sym = get_current_commodity_symbol("CRUDEOIL")
        if crude_sym:
            COMMODITY_SYMBOLS["CRUDE"] = crude_sym
            updated = True
        if updated:
            _rebuild_instrument_maps()
    except Exception:
        return


def _simulate_trades_from_signals(df: pd.DataFrame, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if df.empty or not signals:
        return []

    candles = df.reset_index(drop=True)
    times = candles["date"]
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    closes = candles["close"].to_numpy(dtype=float)

    trades: List[Dict[str, Any]] = []
    for s in signals:
        try:
            action = str(s.get("action") or "").upper()
            if action not in {"BUY", "SELL"}:
                continue

            entry_price = float(s.get("entry_price"))
            stop_loss = float(s.get("stop_loss"))
            target = float(s.get("target"))
            entry_time_raw = s.get("timestamp")
            entry_time = pd.to_datetime(entry_time_raw) if entry_time_raw is not None else None
            if entry_time is None or pd.isna(entry_time):
                continue

            if getattr(entry_time, "tzinfo", None) is None:
                try:
                    entry_time = entry_time.tz_localize(IST)  # type: ignore[union-attr]
                except Exception:
                    entry_time = IST.localize(entry_time.to_pydatetime())
            else:
                try:
                    entry_time = entry_time.tz_convert(IST)  # type: ignore[union-attr]
                except Exception:
                    pass

            idx_arr = candles.index[times >= entry_time]
            if len(idx_arr) == 0:
                continue
            entry_idx = int(idx_arr[0])

            exit_idx: Optional[int] = None
            exit_reason = ""
            exit_price = float(closes[min(entry_idx, len(closes) - 1)])

            for j in range(entry_idx + 1, len(candles)):
                hi = float(highs[j])
                lo = float(lows[j])

                if action == "BUY":
                    hit_sl = lo <= stop_loss
                    hit_tgt = hi >= target
                    if hit_sl and hit_tgt:
                        exit_idx = j
                        exit_price = float(stop_loss)
                        exit_reason = "StopLoss"
                        break
                    if hit_sl:
                        exit_idx = j
                        exit_price = float(stop_loss)
                        exit_reason = "StopLoss"
                        break
                    if hit_tgt:
                        exit_idx = j
                        exit_price = float(target)
                        exit_reason = "Target"
                        break
                else:
                    hit_sl = hi >= stop_loss
                    hit_tgt = lo <= target
                    if hit_sl and hit_tgt:
                        exit_idx = j
                        exit_price = float(stop_loss)
                        exit_reason = "StopLoss"
                        break
                    if hit_sl:
                        exit_idx = j
                        exit_price = float(stop_loss)
                        exit_reason = "StopLoss"
                        break
                    if hit_tgt:
                        exit_idx = j
                        exit_price = float(target)
                        exit_reason = "Target"
                        break

            is_open = False
            if exit_idx is None:
                exit_idx = len(candles) - 1
                exit_price = float(closes[exit_idx])
                exit_reason = "EOD"
                # If this is the absolute very last candle of the dataset, it's an OPEN trade
                is_open = True

            exit_time = times.iloc[exit_idx]
            try:
                if getattr(exit_time, "tzinfo", None) is None:
                    exit_time = exit_time.tz_localize(IST)  # type: ignore[union-attr]
                else:
                    exit_time = exit_time.tz_convert(IST)  # type: ignore[union-attr]
            except Exception:
                pass

            if action == "BUY":
                points = exit_price - entry_price
            else:
                points = entry_price - exit_price

            result = "Win" if points > 0 else "Loss"
            duration = exit_time - entry_time

            trades.append(
                {
                    "DateTime": _to_ist_iso(entry_time),
                    "Direction": "B" if action == "BUY" else "S",
                    "Entry Price": float(entry_price),
                    "Entry Time": _to_ist_iso(entry_time),
                    "Exit Price": float(exit_price),
                    "Exit Time": _to_ist_iso(exit_time),
                    "Stoploss": float(stop_loss),
                    "Target": float(target),
                    "Points Captured": float(points),
                    "Result": result,
                    "Trade Duration": str(duration),
                    "Exit Reason": exit_reason,
                    "Status": "OPEN" if is_open else "CLOSED",
                    "Unrealized P&L": float(points) if is_open else 0.0,
                }
            )
        except Exception:
            continue

    return trades


def _compute_summary(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_profit_inr": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "winning_streak": 0,
            "losing_streak": 0,
            "is_profitable": False,
            "setup_reliability_score": 0.0,
        }

    points = pd.Series([float(t.get("Points Captured") or 0.0) for t in trades])
    wins = points[points > 0]
    losses = points[points <= 0]

    total_trades = int(len(points))
    total_wins = int(len(wins))
    total_losses = int(len(losses))
    win_rate = (total_wins / total_trades) * 100.0 if total_trades else 0.0

    gross_profit = float(wins.sum())
    gross_loss = float((-losses).sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity = points.cumsum()
    running_max = equity.cummax()
    dd = running_max - equity
    max_dd = float(dd.max()) if not dd.empty else 0.0

    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for p in points.tolist():
        if p > 0:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    total_profit = float(points.sum())
    is_profitable = total_profit > 0

    pf_score = 1.0 if profit_factor == float("inf") else float(min(profit_factor / 2.0, 1.0))
    win_score = float(min(win_rate / 60.0, 1.0))
    reliability = float(round((pf_score * 0.6 + win_score * 0.4) * 100.0, 2))

    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "total_profit_inr": round(total_profit, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "winning_streak": int(max_win_streak),
        "losing_streak": int(max_loss_streak),
        "is_profitable": bool(is_profitable),
        "setup_reliability_score": reliability,
    }


def _format_backtest_trade_log(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, t in enumerate(trades, start=1):
        exit_reason = str(t.get("Exit Reason") or "")
        strategy = str(t.get("Strategy") or "")
        timeframe = str(t.get("Timeframe") or "")
        status = t.get("Status", "CLOSED")

        is_open = status == "OPEN"

        if is_open:
            outcome = ""
            exit_time_display = "LIVE / OPEN"
            exit_price_str = f"({t.get('Exit Price', '')})"
            realized_pnl = ""
            unrealized_pnl = t.get("Points Captured") or ""
        else:
            if exit_reason.lower().startswith("stop"):
                outcome = "SL Hit"
            elif exit_reason.lower().startswith("target"):
                outcome = "TGT Hit"
            else:
                outcome = ""
            exit_time_display = t.get("Exit Time") or ""
            exit_price_str = str(t.get("Exit Price", ""))
            realized_pnl = t.get("Points Captured") or ""
            unrealized_pnl = ""

        entry_time = t.get("Entry Time")

        reason_for_trade = strategy
        if strategy and timeframe:
            reason_for_trade = f"{strategy} ({timeframe})"
        elif not reason_for_trade:
            reason_for_trade = exit_reason

        out.append(
            {
                "Sr. No.": idx,
                "Status": status,
                "Symbol": t.get("Symbol") or "",
                "Direction (Buy/Sell)": "BUY" if str(t.get("Direction") or "").upper() in {"B", "BUY"} else "SELL",
                "Entry Price": t.get("Entry Price") or "",
                "Entry Time (IST)": entry_time or "",
                "Exit Price": exit_price_str,
                "Exit Time (IST)": exit_time_display,
                "Points Captured (Realized)": realized_pnl,
                "Unrealized P&L": unrealized_pnl,
                "Reason for Trade": reason_for_trade,
                "Outcome": outcome,
            }
        )
    return out


def _make_xlsx_bytes(trades: List[Dict[str, Any]], summary: Dict[str, Any]) -> bytes:
    trade_cols = [
        "Sr. No.",
        "Status",
        "Symbol",
        "Direction (Buy/Sell)",
        "Entry Price",
        "Entry Time (IST)",
        "Exit Price",
        "Exit Time (IST)",
        "Points Captured (Realized)",
        "Unrealized P&L",
        "Reason for Trade",
        "Outcome",
    ]
    df_trades = pd.DataFrame(_format_backtest_trade_log(trades))
    if not df_trades.empty:
        for c in trade_cols:
            if c not in df_trades.columns:
                df_trades[c] = ""
        df_trades = df_trades[trade_cols]
    else:
        df_trades = pd.DataFrame(
            [
                {
                    "Sr. No.": "",
                    "Status": "",
                    "Symbol": "DEBUG: NO TRADES FOUND IN DATASET",
                    "Direction (Buy/Sell)": "",
                    "Entry Price": "",
                    "Entry Time (IST)": "",
                    "Exit Price": "",
                    "Exit Time (IST)": "",
                    "Points Captured (Realized)": "",
                    "Unrealized P&L": "",
                    "Reason for Trade": "",
                    "Outcome": "",
                }
            ],
            columns=trade_cols,
        )

    summary_cols = [
        "Total Trades",
        "Total Wins",
        "Total Losses",
        "Win Rate (%)",
        "Profit Factor",
        "Total Profit (INR)",
        "Avg Win",
        "Avg Loss",
        "Max Drawdown",
        "Winning Streaks",
        "Losing Streaks",
        "Is Profitable?",
        "Setup Reliability Score",
    ]
    summary_row = {
        "Total Trades": summary.get("total_trades"),
        "Total Wins": summary.get("total_wins"),
        "Total Losses": summary.get("total_losses"),
        "Win Rate (%)": summary.get("win_rate_pct"),
        "Profit Factor": summary.get("profit_factor"),
        "Total Profit (INR)": summary.get("total_profit_inr"),
        "Avg Win": summary.get("avg_win"),
        "Avg Loss": summary.get("avg_loss"),
        "Max Drawdown": summary.get("max_drawdown"),
        "Winning Streaks": summary.get("winning_streak"),
        "Losing Streaks": summary.get("losing_streak"),
        "Is Profitable?": summary.get("is_profitable"),
        "Setup Reliability Score": summary.get("setup_reliability_score"),
    }
    df_summary = pd.DataFrame([summary_row], columns=summary_cols)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_trades.to_excel(writer, index=False, sheet_name="Trade Log")
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
    return buf.getvalue()


def _run_backtest_job(job_id: str, payload: Dict[str, Any]) -> None:
    segment = str(payload.get("segment") or "EQUITY").upper()
    symbol_sel = str(payload.get("symbol") or "ALL").upper()
    timeframe_sel = str(payload.get("timeframe") or "ALL")
    strategy_sel = str(payload.get("strategy") or "ALL")
    date_from = _to_yyyy_mm_dd(str(payload.get("date_from") or ""))
    date_to = _to_yyyy_mm_dd(str(payload.get("date_to") or ""))

    runner = LiveStrategyRunner()
    all_strategies = list(runner.strategies.keys())

    symbols: List[str]
    if segment == "MCX":
        _resolve_mcx_active_contracts()
        base = ["CRUDE", "GOLD", "SILVER"]
        symbols = base if symbol_sel == "ALL" else [symbol_sel]
    else:
        base = list(NIFTY_50_SYMBOLS.keys())
        symbols = base if symbol_sel == "ALL" else [symbol_sel]

    timeframes: List[str]
    if str(timeframe_sel).upper() == "ALL":
        # Use a stable set of canonical labels for the engine/strategies.
        timeframes = ["1m", "3m", "15m", "30m", "60m", "120m", "240m", "1D"]
    else:
        timeframes = [_normalize_backtest_timeframe(str(timeframe_sel))]

    strategies: List[str]
    if str(strategy_sel).upper() == "ALL":
        strategies = all_strategies
    else:
        strategies = [str(strategy_sel)]

    tasks: List[Tuple[str, str, str]] = []
    for sym in symbols:
        for tf in timeframes:
            for st in strategies:
                tasks.append((sym, tf, st))

    total_tasks = len(tasks)
    completed_tasks = 0
    trades_all: List[Dict[str, Any]] = []

    with _backtest_jobs_lock:
        j = _backtest_jobs.get(job_id)
        if j is not None:
            j["debug"] = {
                "history_attempts": 0,
                "history_failures": 0,
                "last_failed": None,
                "total_candles": 0,
                "ema_cross_signals": 0,
            }

    for sym, tf, st in tasks:
        completed_tasks += 1
        with _backtest_jobs_lock:
            job = _backtest_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["message"] = f"Running {sym} {tf} {st}"
            job["progress_pct"] = round((completed_tasks / max(total_tasks, 1)) * 100.0, 2)

        code = _resolve_backtest_instrument(segment, sym)

        with _backtest_jobs_lock:
            j = _backtest_jobs.get(job_id)
            if j is not None and isinstance(j.get("debug"), dict):
                j["debug"]["history_attempts"] = int(j["debug"].get("history_attempts", 0)) + 1

        tf_norm = _normalize_backtest_timeframe(tf)
        df, raw_resp = _fetch_history_df(code, tf_norm, date_from, date_to, display_symbol=sym)
        time.sleep(0.6)
        if df.empty:
            with _backtest_jobs_lock:
                j = _backtest_jobs.get(job_id)
                if j is not None and isinstance(j.get("debug"), dict):
                    j["debug"]["history_failures"] = int(j["debug"].get("history_failures", 0)) + 1
                    j["debug"]["last_failed"] = {
                        "symbol": sym,
                        "timeframe": tf,
                        "instrument": code,
                        "response": raw_resp,
                    }
            continue

        with _backtest_jobs_lock:
            j = _backtest_jobs.get(job_id)
            if j is not None and isinstance(j.get("debug"), dict):
                j["debug"]["total_candles"] = int(j["debug"].get("total_candles", 0)) + int(len(df))

        if st == "ema_crossover_5_20":
            try:
                df_dbg = df.copy()
                df_dbg["ema5"] = df_dbg["close"].ewm(span=5, adjust=False).mean()
                df_dbg["ema20"] = df_dbg["close"].ewm(span=20, adjust=False).mean()
                cross_up = (df_dbg["ema5"] > df_dbg["ema20"]) & (df_dbg["ema5"].shift(1) <= df_dbg["ema20"].shift(1))
                cross_down = (df_dbg["ema5"] < df_dbg["ema20"]) & (df_dbg["ema5"].shift(1) >= df_dbg["ema20"].shift(1))
                df_dbg["signal"] = 0
                df_dbg.loc[cross_up, "signal"] = 1
                df_dbg.loc[cross_down, "signal"] = -1
                trades_found = int((df_dbg["signal"] != 0).sum())
                print(f"Symbol: {sym}, Strategy 5: {trades_found} trades found.")

                with _backtest_jobs_lock:
                    j = _backtest_jobs.get(job_id)
                    if j is not None and isinstance(j.get("debug"), dict):
                        j["debug"]["ema_cross_signals"] = int(j["debug"].get("ema_cross_signals", 0)) + trades_found
            except Exception as e:
                print(f"Symbol: {sym}, Strategy 5 verification failed: {e}")

        df_local = df.copy()
        df_local = runner._ensure_required_columns(df_local)
        strat = runner.strategies.get(st)
        if strat is None:
            continue

        htf_tf = "60m"
        try:
            htf_tf = "60m" if _tf_minutes(tf_norm) < 60 else tf_norm
        except Exception:
            htf_tf = "60m"

        htf_df_bt = None
        try:
            if htf_tf != tf_norm:
                htf_df_bt, _ = _fetch_history_df(code, htf_tf, date_from, date_to, display_symbol=sym)
        except Exception:
            htf_df_bt = None

        signals = strat.run_strategy(df_local, sym, tf_norm)
        sig_dicts: List[Dict[str, Any]] = []
        for s in signals:
            # Bypass SMC validation for simple EMA scalping as it follows different rules
            if st == "ema_scalping_5":
                sig_dicts.append(
                    {
                        "action": s.action,
                        "entry_price": float(s.entry_price),
                        "stop_loss": float(s.stop_loss),
                        "target": float(s.target),
                        "timestamp": s.timestamp,
                    }
                )
                continue

            smc = is_smc_valid(
                code,
                s.action,
                float(s.entry_price),
                ltf_df=df_local,
                htf_df=htf_df_bt,
                ltf_timeframe=tf_norm,
                htf_timeframe=htf_tf,
            )
            if not smc.valid:
                continue

            entry_adj = float(s.entry_price)
            if smc.order_type == "LIMIT" and smc.limit_price is not None:
                entry_adj = float(smc.limit_price)
            sl_adj = float(s.stop_loss)
            if smc.stop_loss is not None:
                sl_adj = float(smc.stop_loss)
            tgt_adj = float(s.target)
            if smc.take_profit is not None:
                tgt_adj = float(smc.take_profit)

            sig_dicts.append(
                {
                    "action": s.action,
                    "entry_price": entry_adj,
                    "stop_loss": sl_adj,
                    "target": tgt_adj,
                    "timestamp": s.timestamp,
                }
            )

        trades = _simulate_trades_from_signals(df_local, sig_dicts)
        for t in trades:
            t["Symbol"] = sym
            t["Timeframe"] = tf_norm
            t["Strategy"] = st
        trades_all.extend(trades)

    summary = _compute_summary(trades_all)
    xlsx_bytes = _make_xlsx_bytes(trades_all, summary)
    trade_log_rows = _format_backtest_trade_log(trades_all)

    with _backtest_jobs_lock:
        job = _backtest_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "completed"
        job["message"] = "Completed"
        job["progress_pct"] = 100.0
        job["summary"] = summary
        job["trades_raw"] = trades_all
        job["trades"] = trade_log_rows
        job["xlsx"] = xlsx_bytes


class DashboardWebSocketManager:
    """Lightweight wrapper around FyersDataSocket for dashboard live ticks.

    Tick processing is decoupled: onmessage only enqueues raw messages so the
    Fyers receive thread is never blocked by heavy strategy/candle logic.
    A separate background thread (_tick_processor_worker) drains the queue.
    """

    def __init__(self, instrument_codes: List[str]):
        self.instrument_codes = instrument_codes
        self.fyers_ws: Optional[Any] = None
        self._connected = threading.Event()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Fyers WebSocket callbacks
    # ------------------------------------------------------------------

    def onmessage(self, message: Any) -> None:  # pragma: no cover - I/O heavy
        """Receive tick from Fyers and immediately enqueue it.

        This MUST stay fast — no heavy logic here.  The worker thread does
        the real work so the data-receive thread is never blocked.
        """
        global _last_tick_time, _total_ticks_today, _ticks_today_date

        if not isinstance(message, dict):
            return
        if "ltp" not in message:
            return

        symbol_code = message.get("symbol")
        if not symbol_code or symbol_code not in self.instrument_codes:
            return

        # ── Health bookkeeping (cheap, lock-guarded) ──────────────────
        now = time.time()
        today_ist = datetime.now(IST).date()
        with _tick_count_lock:
            if _ticks_today_date != today_ist:
                _ticks_today_date = today_ist
                _total_ticks_today = 0
            _total_ticks_today += 1
            _last_tick_time = now

        # ── Enqueue for worker thread (non-blocking drop if full) ─────
        try:
            _tick_queue.put_nowait(message)
        except queue.Full:
            pass  # Prefer dropping a tick over blocking the WS receive thread

    def onerror(self, message: Any) -> None:  # pragma: no cover - logging only
        logger.error("Dashboard WebSocket error: %s", message)

    def onclose(self, message: Any) -> None:  # pragma: no cover - logging only
        logger.warning("Connection Closed — Dashboard WebSocket lost: %s", message)
        self._connected.clear()

    def onopen(self) -> None:  # pragma: no cover - logging only
        logger.info("Dashboard WebSocket opened, subscribing to %d symbols", len(self.instrument_codes))
        try:
            self._connected.set()
            if self.fyers_ws is None:
                return
            self.fyers_ws.subscribe(
                symbols=self.instrument_codes,
                data_type="SymbolUpdate",
                channel=15,
            )
            self.fyers_ws.keep_running()
            # Catch up on any missed candles since we went offline
            _catch_up_last_candles()
        except Exception as e:
            logger.error("Error in Dashboard WebSocket onopen: %s", e)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Gracefully shut down the Fyers WebSocket connection."""
        self._connected.clear()
        ws = self.fyers_ws
        self.fyers_ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        logger.info("Dashboard WebSocket stopped")

    def start(self) -> None:
        """Start the FyersDataSocket connection in a background thread."""

        if data_ws is None:
            logger.warning("FyersDataSocket SDK not available; skipping dashboard WebSocket feed")
            return

        # Already running
        if self.fyers_ws is not None:
            return

        auth = _get_fyers_ws_auth()
        if auth is None:
            return

        _app_id, websocket_token = auth

        try:
            self.fyers_ws = data_ws.FyersDataSocket(  # type: ignore[operator]
                access_token=websocket_token,
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self.onopen,
                on_close=self.onclose,
                on_error=self.onerror,
                on_message=self.onmessage,
            )

            t = threading.Thread(target=self.fyers_ws.connect, daemon=True, name="fyers-ws-recv")
            t.start()
            logger.info("Dashboard WebSocket started (reconnect=True)")
        except Exception as e:  # pragma: no cover - network heavy
            logger.error("Error starting Dashboard Fyers WebSocket: %s", e)
            self.fyers_ws = None

    def restart(self) -> None:
        """Force-stop then re-start the WebSocket connection."""
        logger.info("DashboardWebSocketManager.restart() called")
        self.stop()
        time.sleep(1.5)
        self.start()

    def is_connected(self) -> bool:
        return self._connected.is_set()


# ---------------------------------------------------------------------------
# Tick-processing worker: drains _tick_queue and updates caches / candles
# ---------------------------------------------------------------------------

def _process_single_tick(message: Dict[str, Any]) -> None:
    """Process one raw tick from the queue (runs in worker thread)."""
    try:
        symbol_code = message.get("symbol", "")
        ltp_val = float(message.get("ltp") or 0.0)
        volume_val = float(message.get("volume") or 0.0)

        ts_val = message.get("timestamp")
        if ts_val is not None:
            try:
                ts = datetime.fromtimestamp(ts_val, IST)
            except Exception:
                ts = datetime.now(IST)
        else:
            ts = datetime.now(IST)

        display_symbol = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)

        # Update last crude tick time for the 3-minute specific watchdog
        if "CRUDEOIL" in symbol_code.upper():
            global _last_crude_tick_time
            _last_crude_tick_time = time.time()

        # Update live candles for the chart
        try:
            cm = _ensure_candle_manager()
            if cm is not None:
                ts_raw = (
                    message.get("timestamp")
                    or message.get("exch_feed_time")
                    or message.get("last_traded_time")
                    or time.time()
                )
                if not isinstance(ts_raw, (int, float)):
                    ts_raw = time.time()

                tick_volume = message.get("last_traded_qty") or message.get("volume") or 0

                with _candle_lock:
                    cm.update_tick(
                        display_symbol,
                        {
                            "timestamp": ts_raw,
                            "price": ltp_val,
                            "volume": int(tick_volume or 0),
                        },
                    )
        except Exception as e:
            logger.debug("Live candle update error: %s", e)

        # Route tick to the correct feed cache
        is_commodity = symbol_code.startswith("MCX:") or display_symbol in COMMODITY_SYMBOLS

        if is_commodity:
            with _commodity_feed_lock:
                existing = _commodity_feed_cache.get(display_symbol) or {}
                existing.update(
                    {
                        "symbol": display_symbol,
                        "raw_symbol": symbol_code,
                        "ltp": ltp_val,
                        "volume": volume_val,
                        "change_pct": float(existing.get("change_pct", 0.0)),
                        "updated_at": ts.strftime("%d-%b-%Y %I:%M %p"),
                        "updated_epoch": time.time(),
                    }
                )
                _commodity_feed_cache[display_symbol] = existing
        else:
            with _market_feed_lock:
                existing = _market_feed_cache.get(display_symbol) or {}
                existing.update(
                    {
                        "symbol": display_symbol,
                        "raw_symbol": symbol_code,
                        "ltp": ltp_val,
                        "volume": volume_val,
                        "change_pct": float(existing.get("change_pct", 0.0)),
                        "updated_at": ts.strftime("%d-%b-%Y %I:%M %p"),
                        "updated_epoch": time.time(),
                    }
                )
                _market_feed_cache[display_symbol] = existing

    except Exception as e:
        logger.error("Tick processing error: %s", e)


def _tick_processor_worker() -> None:
    """Background thread: drains the tick queue and calls _process_single_tick.

    Separating data receipt (WebSocket thread) from data processing (this thread)
    means heavy candle/cache logic never blocks the Fyers socket receive loop.
    """
    logger.info("Tick processor worker thread started")
    while True:
        try:
            msg = _tick_queue.get(timeout=1.0)
            _process_single_tick(msg)
            _tick_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Tick processor worker error: %s", e)


# ---------------------------------------------------------------------------
# WebSocket Watchdog: forces reconnect if no ticks arrive for 60s
# ---------------------------------------------------------------------------

_WS_WATCHDOG_TIMEOUT_S = 60    # reconnect if no tick for this many seconds
_WS_HEARTBEAT_INTERVAL_S = 30  # log a heartbeat ping every N seconds


def _ws_watchdog_fn() -> None:
    """Watchdog loop: monitors tick health and triggers forced reconnect.

    Runs in a daemon thread started at app startup.
    """
    global _last_tick_time, _fyers_token_timestamp, _last_crude_tick_time
    logger.info("WS watchdog thread started (timeout=%ds, heartbeat_interval=%ds)",
                _WS_WATCHDOG_TIMEOUT_S, _WS_HEARTBEAT_INTERVAL_S)

    last_heartbeat_log = time.time()

    while True:
        try:
            time.sleep(5)
            now = time.time()
            now_dt = datetime.fromtimestamp(now, IST)

            # 1. ── Session Refresh / Auto-Login Check (08:45 AM IST) ────────
            # If token is > 24h old and it's around 8:45 AM, we should ideally refresh.
            # For this implementation, we log a critical alert or attempt re-init.
            if now_dt.hour == 8 and now_dt.minute == 45 and (now - _fyers_token_timestamp) > 80000:
                logger.info("Watchdog: Daily session refresh window (08:45 AM). Re-initializing Fyers client...")
                try:
                    _init_fyers_client()
                    _fyers_token_timestamp = now
                except Exception as e:
                    logger.error("Watchdog: Failed daily session refresh: %s", e)

            # 2. ── Periodic heartbeat log ───────────────────────────────────
            if now - last_heartbeat_log >= _WS_HEARTBEAT_INTERVAL_S:
                queue_depth = _tick_queue.qsize()
                elapsed_since_tick = now - _last_tick_time if _last_tick_time > 0 else -1
                elapsed_crude = now - _last_crude_tick_time if _last_crude_tick_time > 0 else -1
                logger.info(
                    "WS Heartbeat ♥ | last_tick=%.0fs | crude_tick=%.0fs | q=%d | ticks=%d",
                    elapsed_since_tick,
                    elapsed_crude,
                    queue_depth,
                    _total_ticks_today,
                )
                last_heartbeat_log = now

            # 3. ── Market Hours Watchdog ────────────────────────────────────
            if not (_is_commodity_market_open() or _is_equity_market_open()):
                continue

            # Standard watchdog (any tick) - 60s
            elapsed = now - _last_tick_time
            if _last_tick_time > 0 and elapsed > _WS_WATCHDOG_TIMEOUT_S:
                logger.warning("Watchdog: No tick (any) for %.0fs — reconnecting", elapsed)
                _force_reconnect_ws()
                with _tick_count_lock: _last_tick_time = now
                continue

            # Crude Oil Specific Watchdog - 3 minutes (180s)
            elapsed_crude = now - _last_crude_tick_time
            if _last_crude_tick_time > 0 and elapsed_crude > 180:
                logger.warning("Watchdog: Crude Oil stalled for %.0fs — reconnecting", elapsed_crude)
                _force_reconnect_ws()
                _last_crude_tick_time = now

        except Exception as e:
            logger.error("WS watchdog error: %s", e)


def _force_reconnect_ws() -> None:
    """Force-close and restart the Fyers DataSocket."""
    global _ws_manager
    with _ws_lock:
        mgr = _ws_manager
    if mgr is None:
        logger.warning("_force_reconnect_ws: no WS manager to reconnect")
        return
    logger.info("Force-reconnecting WebSocket …")
    mgr.restart()


def _catch_up_last_candles() -> None:
    """Fetch the last 3 candles for 1m/5m/15m on MCX Big 3 after reconnect.

    This prevents missed signals when the socket was briefly offline.
    """
    MCX_TARGETS = list(COMMODITY_SYMBOLS.values())  # e.g. ["MCX:GOLDM...", ...]
    TIMEFRAMES = ["1m", "5m", "15m"]

    for sym in MCX_TARGETS:
        for tf in TIMEFRAMES:
            try:
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                df, _ = _fetch_history_df(sym, tf, today_str, today_str, display_symbol=sym)
                if not df.empty:
                    # Just fetching refreshes the candle history; actual signal
                    # processing happens in _paper_engine_loop as normal.
                    logger.debug("Catch-up: fetched %d %s candles for %s", len(df), tf, sym)
            except Exception as e:
                logger.debug("Catch-up error for %s %s: %s", sym, tf, e)


def _init_ws_feed() -> None:
    """Initialise WebSocket-based live tick feed for the dashboard watchlist."""

    global _ws_manager

    if data_ws is None:
        # SDK is optional; if it's missing we silently run without WS.
        logger.info("Fyers WebSocket SDK not available; dashboard will use REST quotes only")
        return

    with _ws_lock:
        if _ws_manager is not None:
            return

        try:
            # Include both NIFTY-50 and commodity symbols for real-time updates
            instrument_codes = list(NIFTY_50_SYMBOLS.values())
            if COMMODITY_SYMBOLS:
                instrument_codes += list(COMMODITY_SYMBOLS.values())
            
            _ws_manager = DashboardWebSocketManager(instrument_codes)
            _ws_manager.start()
            logger.info("Dashboard WebSocket feed initialised for %d symbols (equities + commodities)", len(instrument_codes))
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Error initialising dashboard WebSocket feed: %s", e)
            _ws_manager = None


def _ensure_candle_manager() -> Optional[LiveCandleManager]:
    """Initialise and return the shared live candle manager for the chart."""

    global _candle_manager
    if _candle_manager is not None:
        return _candle_manager

    try:
        symbols = list(NIFTY_50_SYMBOLS.keys()) + list(COMMODITY_SYMBOLS.keys())
        timeframes = ["1m", "5m", "15m"]
        _candle_manager = LiveCandleManager(symbols, timeframes)
        logger.info("Live candle manager initialised for %d symbols", len(symbols))
        return _candle_manager
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Error initialising live candle manager: %s", e)
        _candle_manager = None
        return None


def _get_historical_candles(symbol: str, timeframe: str = "5m", count: int = 200) -> pd.DataFrame:
    """Fetch historical candles from Fyers for a symbol/timeframe.

    All data comes directly from the broker's history() API; no simulation
    or synthetic candles are generated here.
    """

    if _fyers_client is None:
        return pd.DataFrame()

    tf_mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "60m": "60",
        "120m": "120",
        "180m": "180",
        "240m": "240",
        "1D": "D",
    }

    fyers_tf = tf_mapping.get(timeframe, timeframe)

    end_dt = datetime.now(IST)
    if timeframe == "1D":
        start_dt = end_dt - timedelta(days=count)
    else:
        tf_minutes = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "60m": 60,
            "120m": 120,
            "180m": 180,
            "240m": 240,
        }
        minutes_needed = count * tf_minutes.get(timeframe, 60)
        start_dt = end_dt - timedelta(minutes=minutes_needed)

    # Resolve display symbol (e.g. "RELIANCE" or "GOLD") to full
    # instrument code using the combined symbol map that includes both
    # equities and commodities. Fall back to NSE prefix if unknown.
    code = _ALL_SYMBOLS_MAP.get(symbol, symbol)
    if ":" not in code:
        code = f"NSE:{code}"

    data = {
        "symbol": code,
        "resolution": fyers_tf,
        "date_format": "1",
        "range_from": start_dt.strftime("%Y-%m-%d"),
        "range_to": end_dt.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    try:
        resp = _fyers_client.history(data)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error fetching historical candles for {symbol}: {e}")
        return pd.DataFrame()

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        logger.warning(f"Unexpected history response for {symbol}: {resp}")
        return pd.DataFrame()

    candles = resp.get("candles") or []
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
    return df[["date", "open", "high", "low", "close", "volume"]].tail(count)


def _get_live_candles(symbol: str, timeframe: str = "5m", count: int = 200) -> pd.DataFrame:
    """Return live candles built from WebSocket ticks for a symbol/timeframe."""

    cm = _ensure_candle_manager()
    if cm is None:
        return pd.DataFrame()

    try:
        with _candle_lock:
            return cm.get_candles(symbol, timeframe, count)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error fetching live candles for {symbol} {timeframe}: {e}")
        return pd.DataFrame()


def _make_signal_id(signal: Dict[str, Any]) -> str:
    symbol = str(signal.get("symbol") or "")
    timeframe = str(signal.get("signal_timeframe") or signal.get("timeframe") or "")
    action = str(signal.get("final_action") or signal.get("action") or "")
    ts = str(signal.get("signal_timestamp") or signal.get("timestamp_generated") or "")
    return "|".join([symbol, timeframe, action, ts])


def _normalize_instrument(symbol: str) -> str:
    sym = str(symbol or "").strip()
    if not sym:
        return sym

    if sym in INSTRUMENT_TO_DISPLAY:
        normalized = sym
        logger.info("PaperExec: DEBUG normalize input=%r normalized=%r (direct instrument)", sym, normalized)
        return normalized

    if sym in NIFTY_50_SYMBOLS:
        normalized = NIFTY_50_SYMBOLS[sym]
        logger.info("PaperExec: DEBUG normalize input=%r normalized=%r (nifty50 display)", sym, normalized)
        return normalized

    for inst, disp in INSTRUMENT_TO_DISPLAY.items():
        if disp == sym:
            logger.info("PaperExec: DEBUG normalize input=%r normalized=%r (display->inst)", sym, inst)
            return inst
        if inst.endswith(sym):
            logger.info("PaperExec: DEBUG normalize input=%r normalized=%r (suffix match)", sym, inst)
            return inst

    logger.info("PaperExec: DEBUG normalize input=%r normalized=%r (fallback passthrough)", sym, sym)
    return sym


def _get_latest_ltp(symbol_code: str) -> Optional[float]:
    display_symbol = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)

    source = "none"
    with _market_feed_lock:
        row = _market_feed_cache.get(display_symbol)
    if row:
        source = "market"
    else:
        with _commodity_feed_lock:
            row = _commodity_feed_cache.get(display_symbol)
        if row:
            source = "commodity"

    if not row:
        return None

    value = row.get("ltp")
    if value is None:
        return None

    try:
        ltp = float(value)
    except Exception:
        return None

    if ltp <= 0:
        logger.info(
            "PaperExec: DEBUG LTP_ZERO symbol_code=%s display_symbol=%s source=%s row=%r",
            symbol_code,
            display_symbol,
            source,
            row,
        )

    return ltp


def _open_paper_trade_from_signal(signal: Dict[str, Any]) -> None:
    global _paper_trades, _paper_positions, _paper_trades_version

    # ── HARD GATE: WebSocket must be live ─────────────────────────────────────
    if not _is_ws_live():
        age = time.time() - _last_tick_time if _last_tick_time > 0 else -1
        logger.warning(
            "PaperExec: BLOCKED – WebSocket is offline/stale (last_tick_age=%.0fs). "
            "No trade will be opened until real-time WS ticks resume.",
            age,
        )
        return

    symbol_raw = str(signal.get("symbol") or "")
    symbol_code = _normalize_instrument(symbol_raw)
    if not symbol_code:
        logger.info(
            "PaperExec: skip signal with empty/unnormalizable symbol %r",
            symbol_raw,
        )
        return

    if not _is_market_open_for_instrument(symbol_code):
        logger.info(
            "PaperExec: skip signal for %s because market is closed (now=%s IST)",
            symbol_code,
            datetime.now(IST).strftime("%d-%b-%Y %I:%M %p"),
        )
        return

    direction = str(signal.get("final_action") or signal.get("action") or "").upper()
    if direction not in {"BUY", "SELL"}:
        logger.info(
            "PaperExec: skip signal for %s due to invalid direction %r",
            symbol_code,
            direction,
        )
        return

    if symbol_code in _paper_positions:
        logger.info(
            "PaperExec: skip signal for %s because an open paper position already exists",
            symbol_code,
        )
        return

    entry_price = signal.get("entry_price")
    stop_loss = signal.get("stop_loss_level")
    target = signal.get("target_level")
    if entry_price is None or stop_loss is None or target is None:
        logger.info(
            "PaperExec: skip signal for %s due to missing prices (entry=%r, sl=%r, tgt=%r)",
            symbol_code,
            entry_price,
            stop_loss,
            target,
        )
        return

    try:
        entry_signal = float(entry_price)
        sl = float(stop_loss)
        tgt = float(target)
    except Exception:
        logger.info(
            "PaperExec: skip signal for %s due to non-numeric prices (entry=%r, sl=%r, tgt=%r)",
            symbol_code,
            entry_price,
            stop_loss,
            target,
        )
        return

    if entry_signal <= 0:
        logger.info(
            "PaperExec: skip signal for %s due to non-positive entry price %r",
            symbol_code,
            entry_signal,
        )
        return

    direction = str(signal.get("final_action") or signal.get("action") or "").upper()
    entry_live = _paper_compute_entry_price(symbol_code, direction, fallback=entry_signal)
    if entry_live is None or entry_live <= 0:
        logger.info(
            "PaperExec: skip signal for %s due to missing/invalid LTP for entry (fallback_entry=%r)",
            symbol_code,
            entry_signal,
        )
        return

    display_symbol = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)

    ts_raw = signal.get("signal_timestamp") or signal.get("timestamp_generated")
    when = datetime.now(IST)
    if isinstance(ts_raw, str):
        try:
            parsed = datetime.fromisoformat(ts_raw)
            if parsed.tzinfo is None:
                parsed = pytz.utc.localize(parsed)
            when = parsed.astimezone(IST)
        except Exception:
            pass

    sym_up = symbol_code.upper()
    if sym_up.startswith("MCX:"):
        qty = 1  # MCX paper trading: 1 lot
        lot_size = _commodity_lot_size(symbol_code)
    else:
        qty = 1  # Equity paper trading: 1 quantity
        lot_size = 1
    timeframe = str(signal.get("signal_timeframe") or signal.get("timeframe") or "")
    confidence_val = signal.get("aggregated_confidence") or signal.get("confidence") or 0.0
    decision_score = signal.get("decision_score") or 0.0

    strategies = signal.get("contributing_strategies")
    if isinstance(strategies, list):
        strategy_str = ", ".join(strategies)
    else:
        strategy_str = str(strategies or "Aggregated")

    trade = {
        "trade_id": str(uuid.uuid4()),
        "symbol": symbol_code,
        "display_symbol": display_symbol,
        "direction": direction,
        "qty": qty,
        "lot_size": lot_size,
        "entry_price": float(entry_live),
        "entry_time": when.strftime("%d-%b-%Y %I:%M %p"),
        "exit_price": None,
        "exit_time": None,
        "status": "OPEN",
        "timeframe": timeframe,
        "strategy": strategy_str,
        "reason": "EngineSignal",
        "stop_loss": sl,
        "target": tgt,
        "aggregated_confidence": float(confidence_val),
        "decision_score": float(decision_score),
    }

    _paper_trades.append(trade)
    _paper_positions[symbol_code] = trade
    _paper_trades_version += 1

    _persist_paper_trades_snapshot(_paper_trades)

    logger.info(
        "PaperExec: OPEN %s %s qty=%s entry=%.4f sl=%.4f tgt=%.4f (display=%s)",
        direction,
        symbol_code,
        qty,
        float(entry_live),
        sl,
        tgt,
        display_symbol,
    )


def _update_paper_positions_from_prices() -> None:
    global _paper_positions, _paper_trades_version

    # ── HARD GATE: Only check exits when WS ticks are live ────────────────────
    if not _is_ws_live():
        age = time.time() - _last_tick_time if _last_tick_time > 0 else -1
        logger.warning(
            "PaperExec: EXIT CHECK BLOCKED – WebSocket offline/stale (last_tick_age=%.0fs). "
            "Holding all positions until WS reconnects.",
            age,
        )
        return

    now = datetime.now(IST).strftime("%d-%b-%Y %I:%M %p")
    to_close: List[str] = []
    changed = False

    for symbol_code, trade in list(_paper_positions.items()):
        if not _is_market_open_for_instrument(symbol_code):
            continue
        entry_price = trade.get("entry_price")
        stop_loss = trade.get("stop_loss")
        target = trade.get("target")
        if entry_price is None or stop_loss is None or target is None:
            continue

        try:
            entry = float(entry_price)
            sl = float(stop_loss)
            tgt = float(target)
        except Exception:
            continue

        if entry <= 0:
            continue

        ltp = _get_latest_ltp(symbol_code)
        if ltp is None or ltp <= 0:
            logger.info(
                "PaperExec: CHECK symbol=%s skipped due to missing/invalid LTP (ltp=%r)",
                symbol_code,
                ltp,
            )
            continue

        direction = str(trade.get("direction") or "BUY").upper()
        lot_size = _commodity_lot_size(symbol_code)
        qty = trade.get("qty") or 1
        entry = float(trade.get("entry_price") or 0)
        
        # Calculate Turnover for charges (Entry + Exit)
        turnover = (entry + ltp) * qty * lot_size
        charges = _estimate_brokerage_and_taxes(symbol_code, turnover)
        
        # Calculate Points and Gross PnL
        pnl_points = (ltp - entry) if direction == "BUY" else (entry - ltp)
        gross_pnl = pnl_points * qty * lot_size
        net_pnl = gross_pnl - charges

        hit = False
        reason = ""

        # Expiry safety: force-close paper positions close to expiry to simulate
        # avoiding devolvement/physical delivery.
        if symbol_code.upper().startswith("MCX:"):
            days = _mcx_days_to_expiry(symbol_code)
            if days is not None and days <= _MCX_EXPIRY_EXIT_DAYS:
                hit = True
                reason = "ExpiryForceClose"

        logger.info(
            "PaperExec: CHECK symbol=%s dir=%s entry=%.4f sl=%.4f tgt=%.4f ltp=%.4f",
            symbol_code,
            direction,
            entry,
            sl,
            tgt,
            ltp,
        )

        if direction == "BUY":
            if ltp >= tgt:
                hit = True
                reason = "TargetHit"
            elif ltp <= sl:
                hit = True
                reason = "StopLossHit"
        else:
            if ltp <= tgt:
                hit = True
                reason = "TargetHit"
            elif ltp >= sl:
                hit = True
                reason = "StopLossHit"

        if not hit:
            continue

        qty = trade.get("qty", 1)
        try:
            qty_val = int(qty)
        except Exception:
            qty_val = 1

        lot_size = trade.get("lot_size")
        try:
            lot_size_val = int(lot_size) if lot_size is not None else _commodity_lot_size(symbol_code)
        except Exception:
            lot_size_val = _commodity_lot_size(symbol_code)

        logger.info(
            "Debug PnL: Symbol=%s, Entry=%.4f, CurrentLTP=%.4f, Direction=%s, Qty=%s",
            symbol_code,
            entry,
            ltp,
            direction,
            qty_val,
        )

        if direction == "BUY":
            pnl_points = ltp - entry
        else:
            pnl_points = entry - ltp

        # Net P&L (INR): ((Exit - Entry) * Lot Size * Lots) - (Brokerage + Taxes)
        charges = _estimate_brokerage_and_taxes(symbol_code, qty_val)
        pnl_value = (pnl_points * float(lot_size_val) * float(qty_val)) - float(charges)

        if hit:
            trade["exit_price"] = float(ltp)
            trade["exit_time"] = now
            trade["status"] = "CLOSED"
            trade["exit_reason"] = reason
            trade["pnl_points"] = float(pnl_points)
            trade["gross_pnl"] = float(gross_pnl)
            trade["charges"] = float(charges)
            trade["pnl"] = float(net_pnl)
            trade["outcome"] = "TGT Hit" if "Target" in reason else "SL Hit"

            logger.info(
                "PaperExec: PNL_CALC symbol=%s dir=%s entry=%.4f exit=%.4f points=%.4f lot_size=%s qty=%s charges=%.4f net_pnl=%.4f",
                symbol_code,
                direction,
                entry,
                float(ltp),
                float(pnl_points),
                lot_size,
                qty,
                float(charges),
                float(net_pnl),
            )

        required_keys = ["symbol", "direction", "entry_price", "exit_price", "pnl"]
        missing_keys = [k for k in required_keys if k not in trade or trade.get(k) is None]
        if missing_keys:
            logger.error(
                "PaperExec: UI_UPDATE_FAILED missing_keys=%s trade=%r",
                missing_keys,
                trade,
            )

        logger.info(
            "PaperExec: EXIT symbol=%s dir=%s reason=%s entry=%.4f ltp=%.4f qty=%s pnl=%.4f",
            symbol_code,
            direction,
            reason,
            entry,
            ltp,
            qty_val,
            pnl_value,
        )

        to_close.append(symbol_code)
        changed = True

    for symbol_code in to_close:
        _paper_positions.pop(symbol_code, None)

    if changed:
        _paper_trades_version += 1
        _persist_paper_trades_snapshot(_paper_trades)


async def _paper_engine_loop() -> None:
    global _paper_engine_running, _paper_last_heartbeat

    _paper_engine_running = True
    while True:
        try:
            now_ts = time.time()
            if now_ts - _paper_last_heartbeat >= 5.0:
                _paper_last_heartbeat = now_ts
                with _paper_lock:
                    active_codes = sorted(_paper_positions.keys())
                active_symbols: List[str] = []
                for code in active_codes:
                    disp = INSTRUMENT_TO_DISPLAY.get(code, code)
                    active_symbols.append(disp)
                heartbeat_time = datetime.now(IST).strftime("%d-%b-%Y %I:%M %p")
                logger.info(
                    "Engine Pulse: %s | Open positions: %s",
                    heartbeat_time,
                    ", ".join(active_symbols) if active_symbols else "<none>",
                )

                # ── State Management: Force-close expired paper contracts ──
                expired_to_close = []
                with _paper_lock:
                    for code, pos in _paper_positions.items():
                        days = _mcx_days_to_expiry(code)
                        if days is not None and days <= 0:
                            expired_to_close.append(code)
                
                for code in expired_to_close:
                    logger.info("PaperExec: Force-closing expired contract %s", code)
                    # We'll rely on the normal exit logic below if it matches, 
                    # but here we can just trigger a manual close if needed.
                    ltp = _get_latest_ltp(code)
                    if ltp:
                        _paper_exit_trade(code, ltp, "Expiry System Exit")

            manager = _engine_manager
            if manager is not None:
                signals = manager.get_signals(limit=200) or []
                logger.info("PaperExec: fetched %d aggregated signals from engine", len(signals))
                with _paper_lock:
                    for signal in signals:
                        sid = _make_signal_id(signal)
                        if sid in _paper_processed_signals:
                            continue
                        symbol_raw = str(signal.get("symbol") or "")
                        symbol_code = _normalize_instrument(symbol_raw)
                        if not symbol_code:
                            continue

                        # ── Signal Recency Filter ───────────────────────────────────────────
                        # The engine may surface historical backtest signals.
                        # Only act on signals generated (wall-clock) within 7 days.
                        ts_gen_raw = signal.get("timestamp_generated") or signal.get("signal_timestamp") or ""
                        if ts_gen_raw:
                            try:
                                ts_gen_parsed = datetime.fromisoformat(str(ts_gen_raw))
                                if ts_gen_parsed.tzinfo is None:
                                    ts_gen_parsed = IST.localize(ts_gen_parsed)
                                else:
                                    ts_gen_parsed = ts_gen_parsed.astimezone(IST)
                                signal_age_days = (datetime.now(IST) - ts_gen_parsed).total_seconds() / 86400
                                if signal_age_days > 7:
                                    logger.info(
                                        "PaperExec: SKIP stale signal for %s — age=%.1f days (timestamp_generated=%s)",
                                        symbol_code,
                                        signal_age_days,
                                        ts_gen_raw,
                                    )
                                    _paper_processed_signals.add(sid)  # Prevent re-checking
                                    continue
                            except Exception:
                                pass  # Can't parse timestamp — allow through

                        sym_up = symbol_code.upper()

                        # Segment gate: Equity only during 09:15–15:30; MCX until 23:30
                        if not _is_market_open_for_instrument(symbol_code):
                            logger.info("PaperExec: skipping signal for %s - market closed", symbol_code)
                            continue

                        # MCX commodity filter: only Crude Oil, Gold Mini, Silver Mini
                        if sym_up.startswith("MCX:"):
                            if not any(x in sym_up for x in ["CRUDEOIL", "GOLDM", "SILVERM"]):
                                logger.info("PaperExec: skipping non-target commodity %s", symbol_code)
                                _paper_processed_signals.add(sid)
                                continue

                        ltp_now = _get_latest_ltp(symbol_code) if symbol_code else None
                        logger.info(
                            "PaperExec: DEBUG price types symbol_code=%s ltp_type=%s entry_type=%s sl_type=%s tgt_type=%s",
                            symbol_code,
                            type(ltp_now).__name__ if ltp_now is not None else None,
                            type(signal.get("entry_price")).__name__ if signal.get("entry_price") is not None else None,
                            type(signal.get("stop_loss_level")).__name__ if signal.get("stop_loss_level") is not None else None,
                            type(signal.get("target_level")).__name__ if signal.get("target_level") is not None else None,
                        )
                        logger.info(
                            "PaperExec: SIGNAL symbol_raw=%r normalized=%s dir=%r entry=%r sl=%r tgt=%r ltp_now=%r",
                            symbol_raw,
                            symbol_code,
                            signal.get("final_action") or signal.get("action"),
                            signal.get("entry_price"),
                            signal.get("stop_loss_level"),
                            signal.get("target_level"),
                            ltp_now,
                        )

                        direction_tmp = str(signal.get("final_action") or signal.get("action") or "").upper()
                        entry_tmp = signal.get("entry_price")
                        try:
                            entry_tmp_f = float(entry_tmp) if entry_tmp is not None else float(ltp_now or 0.0)
                        except Exception:
                            entry_tmp_f = float(ltp_now or 0.0)

                        # Bypass SMC validation for simple EMA scalping strategy
                        contributing = signal.get("contributing_strategies") or []
                        if "ema_scalping_5" in contributing:
                            smc_valid = True
                            smc_delay = False
                        else:
                            smc = is_smc_valid(symbol_code, direction_tmp, entry_tmp_f, ltf_timeframe="5m", htf_timeframe="60m")
                            smc_valid = smc.valid
                            smc_delay = smc.delay

                        if not smc_valid and smc_delay:
                            continue

                        if not smc_valid and not smc_delay:
                            _paper_processed_signals.add(sid)
                            continue

                        _open_paper_trade_from_signal(signal)
                        if symbol_code and symbol_code in _paper_positions:
                            _paper_processed_signals.add(sid)

                    _update_paper_positions_from_prices()
        except Exception as e:
            logger.error("Paper engine loop error: %s", e)

        await asyncio.sleep(1.0)


# -----------------------------------------------------------------------------
# System Reboot & Market Timer
# -----------------------------------------------------------------------------


def get_market_timer_status() -> Dict[str, Any]:
    """Evaluates current IST time to determine signal engine allowances.

    Returns a dict with EQUITY_ACTIVE, MCX_ACTIVE, and the current IST time string.

    Market Session Rules (IST, Mon-Fri only):
      09:15 – 15:30  → Equity + MCX
      15:30 – 23:30  → MCX only
      23:30 – 09:15  → All suspended
    """
    now = datetime.now(IST)
    current_time = now.time()

    status: Dict[str, Any] = {
        "EQUITY_ACTIVE": False,
        "MCX_ACTIVE": False,
        "current_time_ist": now.strftime("%d-%b-%Y %I:%M %p"),
        "session": "CLOSED",
    }

    if now.weekday() >= 5:           # Weekend — all off
        status["session"] = "WEEKEND"
        return status

    eq_open  = dt_time(9, 15)
    eq_close = dt_time(15, 30)
    mcx_close = dt_time(23, 30)

    if eq_open <= current_time <= eq_close:
        status["EQUITY_ACTIVE"] = True
        status["MCX_ACTIVE"]    = True
        status["session"] = "JOINT"
    elif eq_close < current_time <= mcx_close:
        status["EQUITY_ACTIVE"] = False
        status["MCX_ACTIVE"]    = True
        status["session"] = "MCX_ONLY"
    else:
        status["session"] = "CLOSED"

    return status


def system_reboot() -> None:
    """Execute on startup: wipe all trade/PnL artefacts for a 100% clean slate.

    Files erased:
      • trade_log.json          – unified live trade log
      • pnl_history.csv         – historical PnL rows
      • trades/ directory       – daily JSON snapshots

    In-memory state is reset via the global _paper_* variables after this
    function returns (caller responsibility since globals live in module scope).
    """
    files_to_wipe = [
        os.path.join(BASE_DIR, "trade_log.json"),
        os.path.join(BASE_DIR, "pnl_history.csv"),
    ]
    for fp in files_to_wipe:
        try:
            if os.path.exists(fp):
                os.remove(fp)
                logger.info("system_reboot: removed %s", fp)
        except Exception as e:
            logger.warning("system_reboot: could not remove %s: %s", fp, e)

    # Wipe daily trade snapshot directory
    trades_dir = os.path.join(BASE_DIR, "trades")
    if os.path.isdir(trades_dir):
        import shutil
        try:
            shutil.rmtree(trades_dir)
            os.makedirs(trades_dir, exist_ok=True)
            logger.info("system_reboot: trades directory wiped and recreated")
        except Exception as e:
            logger.warning("system_reboot: could not wipe trades dir: %s", e)

    # Seed a clean trade_log.json so API calls never 404
    clean_slate = {
        "Cumulative_PnL": 0.0,
        "Total_Trades": 0,
        "Win_Rate": "0.0%",
        "Active_Positions": 0,
        "Win_Loss_Ratio": "0 / 0",
        "Trades": [],
    }
    try:
        with open(os.path.join(BASE_DIR, "trade_log.json"), "w", encoding="utf-8") as f:
            json.dump(clean_slate, f, indent=4)
    except Exception as e:
        logger.warning("system_reboot: could not write trade_log.json: %s", e)

    logger.info(
        "SYSTEM REBOOT COMPLETE | All trade logs wiped | Ready for new session | %s IST",
        datetime.now(IST).strftime("%d-%b-%Y %I:%M %p"),
    )


# -----------------------------------------------------------------------------
# FastAPI lifecycle hooks
# -----------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize logging, engine, and data providers on app startup."""

    _init_logging()
    logger.info("Starting FastAPI backend for Mini-Simon dashboard")

    # ── Clean Slate: wipe all previous session data ───────────────────────────
    global _paper_trades, _paper_positions, _paper_trades_version, _paper_processed_signals
    try:
        system_reboot()
    except Exception as e:
        logger.error("system_reboot failed: %s", e)

    # Reset all in-memory paper trading state to zeros
    with _paper_lock:
        _paper_trades.clear()
        _paper_positions.clear()
        _paper_trades_version = 0
        _paper_processed_signals.clear()
    logger.info("In-memory paper trading state reset to zero")

    try:
        _init_commodity_symbols()
    except Exception as e:
        logger.error("Error initialising commodity symbols: %s", e)

    # Start trading engine in a background thread (live or sim still uses same logic)
    _init_engine()

    # Initialize Fyers client for live mode
    try:
        _init_fyers_client()
    except Exception as e:
        logger.error(f"Error initializing Fyers client: {e}")

    # Best-effort start of WebSocket tick feed for the Nifty-50 watchlist.
    try:
        _init_ws_feed()
    except Exception as e:
        logger.error(f"Error initializing Fyers WebSocket feed: {e}")

    # ── High-frequency resiliency: start tick worker + watchdog ──────────────
    try:
        tw = threading.Thread(
            target=_tick_processor_worker,
            daemon=True,
            name="tick-processor-worker",
        )
        tw.start()
        logger.info("Tick processor worker started")
    except Exception as e:
        logger.error("Error starting tick processor worker: %s", e)

    try:
        wt = threading.Thread(
            target=_ws_watchdog_fn,
            daemon=True,
            name="ws-watchdog",
        )
        wt.start()
        logger.info("WebSocket watchdog started")
    except Exception as e:
        logger.error("Error starting WebSocket watchdog: %s", e)

    # Warm up non-tick caches (positions / orders / margin). Market feed
    # comes either from WebSocket ticks or REST quotes on demand.
    _refresh_all_live()

    try:
        asyncio.create_task(_paper_engine_loop())
    except Exception as e:
        logger.error("Error starting paper engine loop: %s", e)

    try:
        asyncio.create_task(_live_execution_loop())
    except Exception as e:
        logger.error("Error starting live execution loop: %s", e)


# -----------------------------------------------------------------------------
# Routes - HTML
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Render main trading dashboard page."""

    context = {
        "request": request,
        "mode": _get_mode(),
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/backtesting", response_class=HTMLResponse)
async def backtesting_terminal(request: Request) -> HTMLResponse:
    runner = LiveStrategyRunner()
    strategies = sorted(list(runner.strategies.keys()))
    context = {
        "request": request,
        "nifty50": sorted(list(NIFTY_50_SYMBOLS.keys())),
        "mcx": ["CRUDE", "GOLD", "SILVER"],
        "strategies": strategies,
    }
    return templates.TemplateResponse("backtesting.html", context)


@app.get("/pure", response_class=HTMLResponse)
async def pure_trade_history(request: Request) -> HTMLResponse:
    context = {
        "request": request,
        "mode": _get_mode(),
    }
    return templates.TemplateResponse("pure.html", context)


 # -----------------------------------------------------------------------------
 # Routes - API (JSON)
 # -----------------------------------------------------------------------------


@app.get("/api/pure/performance")
async def api_pure_performance() -> JSONResponse:
    with _paper_lock:
        trades_raw = list(_paper_trades)

    trades = [_trade_to_pure_schema(t, i) for i, t in enumerate(trades_raw)]

    total_trades = len(trades)
    closed = [t for t in trades if t.get("exit_time")]
    winning_trades = len([t for t in closed if float(t.get("pnl") or 0.0) > 0])
    losing_trades = len([t for t in closed if float(t.get("pnl") or 0.0) < 0])
    win_rate = (winning_trades / len(closed) * 100.0) if closed else 0.0
    total_pnl = sum(float(t.get("pnl") or 0.0) for t in trades)

    return JSONResponse(
        {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "timestamp": datetime.now(IST).isoformat(),
        }
    )


@app.get("/api/pure/trades")
async def api_pure_trades(
    limit: int = Query(10000, ge=1, le=100000),
    strategy: str = Query(""),
) -> JSONResponse:
    with _paper_lock:
        trades_raw = list(_paper_trades)

    trades = [_trade_to_pure_schema(t, i) for i, t in enumerate(trades_raw)]

    strategy_norm = str(strategy or "").strip().lower()
    if strategy_norm:
        trades = [t for t in trades if str(t.get("strategy") or "").strip().lower() == strategy_norm]

    def _ts(tr: Dict[str, Any]) -> float:
        dt = _parse_dt_best_effort(tr.get("exit_time")) or _parse_dt_best_effort(tr.get("entry_time"))
        if dt is None:
            return 0.0
        try:
            return dt.timestamp()
        except Exception:
            return 0.0

    trades.sort(key=_ts, reverse=True)
    trades_limited = trades[: int(limit)]

    return JSONResponse(
        {
            "trades": trades_limited,
            "count": len(trades_limited),
            "source_file": "paper_trades",
            "timestamp": datetime.now(IST).isoformat(),
        }
    )


def _serialise_signal_for_api(signal: Any) -> Dict[str, Any]:
    if isinstance(signal, AggregatedSignal):
        base = getattr(signal, "to_dict", None)
        if callable(base):
            raw = base()
        else:
            raw = signal.__dict__
    elif isinstance(signal, dict):
        raw = signal
    else:
        try:
            raw = dict(signal)
        except Exception:
            raw = {"value": str(signal)}

    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, datetime):
            out[key] = _to_ist_iso(value)
        elif isinstance(value, str) and key in {
            "signal_timestamp",
            "timestamp_generated",
            "timestamp",
            "entry_time",
            "exit_time",
        }:
            out[key] = _to_ist_iso(value)
        elif isinstance(value, date):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


@app.get("/api/mode")
async def get_mode() -> JSONResponse:
    """Return current dashboard mode (live/sim)."""

    return JSONResponse({"mode": _get_mode()})


class ModeRequest(BaseModel):  # type: ignore[misc]
    mode: str


class BacktestStartRequest(BaseModel):  # type: ignore[misc]
    segment: str
    symbol: str
    timeframe: str
    strategy: str
    date_from: str
    date_to: str


@app.post("/api/mode")
async def set_mode(payload: ModeRequest) -> JSONResponse:
    """Switch between live and paper trading modes.

    In live mode, the dashboard shows broker account data (positions/orders/margin).
    In paper mode, the dashboard shows paper trading positions and trades.
    Both modes use real market data from Fyers APIs.
    """

    try:
        _set_mode(payload.mode)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Mode is now just a label ("live" vs "paper").
    # Data for both modes always comes from live helpers when available.
    mode = _get_mode()
    _refresh_all_live()

    return JSONResponse({"mode": mode})


@app.post("/api/backtest/start")
async def api_backtest_start(payload: BacktestStartRequest) -> JSONResponse:
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",
        "message": "Queued",
        "progress_pct": 0.0,
        "created_at": datetime.now(IST).isoformat(),
        "summary": None,
        "trades": None,
        "xlsx": None,
        "error": None,
    }

    with _backtest_jobs_lock:
        _backtest_jobs[job_id] = job

    def _runner() -> None:
        try:
            with _backtest_jobs_lock:
                if job_id in _backtest_jobs:
                    _backtest_jobs[job_id]["status"] = "running"
                    _backtest_jobs[job_id]["message"] = "Starting"

            _run_backtest_job(job_id, payload.model_dump())
        except Exception as e:
            with _backtest_jobs_lock:
                j = _backtest_jobs.get(job_id)
                if j is not None:
                    j["status"] = "failed"
                    j["message"] = "Failed"
                    j["error"] = str(e)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/backtest/status/{job_id}")
async def api_backtest_status(job_id: str) -> JSONResponse:
    with _backtest_jobs_lock:
        job = _backtest_jobs.get(job_id)
        if job is None:
            return JSONResponse({"status": "not_found"}, status_code=404)

        trades = job.get("trades") or []
        preview = trades[:50] if isinstance(trades, list) else []

        return JSONResponse(
            {
                "job_id": job_id,
                "status": job.get("status"),
                "message": job.get("message"),
                "progress_pct": job.get("progress_pct", 0.0),
                "summary": job.get("summary"),
                "trade_preview": preview,
                "debug": job.get("debug"),
                "error": job.get("error"),
            }
        )


@app.get("/api/backtest/download/{job_id}")
async def api_backtest_download(job_id: str) -> StreamingResponse:
    try:
        with _backtest_jobs_lock:
            job = _backtest_jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": "job_not_found"}, status_code=404)
            xlsx = job.get("xlsx")
            trades_raw = job.get("trades_raw")
            summary = job.get("summary")

        if not isinstance(xlsx, (bytes, bytearray)):
            if isinstance(trades_raw, list) and isinstance(summary, dict):
                xlsx = _make_xlsx_bytes(trades_raw, summary)
                with _backtest_jobs_lock:
                    j = _backtest_jobs.get(job_id)
                    if j is not None:
                        j["xlsx"] = xlsx

        if not isinstance(xlsx, (bytes, bytearray)):
            return JSONResponse({"error": "xlsx_not_ready"}, status_code=404)

        filename = f"backtest_{job_id}.xlsx"
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
        return StreamingResponse(
            io.BytesIO(xlsx),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except Exception as e:
        logger.error("Backtest download error: %s", e)
        return JSONResponse({"error": "download_failed", "details": str(e)}, status_code=500)


@app.get("/api/signals")
async def api_signals() -> JSONResponse:
    # Cutoff: only show signals whose wall-clock creation time is ≤ 7 days old
    cutoff_dt = datetime.now(IST) - timedelta(days=7)
    stored = _load_stored_signals_history(limit=5000)

    live: List[Dict[str, Any]] = []
    manager = _engine_manager
    if manager is not None:
        try:
            raw_signals = manager.get_signals(limit=200) or []
            for s in raw_signals:
                serialized = _serialise_signal_for_api(s)
                if "symbol" in serialized:
                    symbol_code = serialized["symbol"]
                    serialized["display_symbol"] = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)

                # Standardize keys for UI Signal Panel
                serialized["Entry Price"] = serialized.get("entry_price")
                serialized["Stoploss"] = serialized.get("stop_loss_level")
                serialized["Target Price"] = serialized.get("target_level")
                serialized["Reason for Trade"] = serialized.get("strategy_name") or serialized.get("reason")
                serialized["Direction (Buy/Sell)"] = str(serialized.get("action") or "").upper()

                live.append(serialized)
        except Exception as e:
            logger.error("api_signals live merge error: %s", e)

    merged = _merge_and_dedupe_signals(live, stored)

    # Standardize stored signals
    for s in merged:
        if "Entry Price" not in s:
            s["Entry Price"] = s.get("entry_price")
        if "Stoploss" not in s:
            s["Stoploss"] = s.get("stop_loss_level")
        if "Target Price" not in s:
            s["Target Price"] = s.get("target_level")
        if "Reason for Trade" not in s:
            s["Reason for Trade"] = s.get("strategy_name") or s.get("reason")
        if "Direction (Buy/Sell)" not in s:
            s["Direction (Buy/Sell)"] = str(s.get("action") or "").upper()

    # Final sort by timestamp_generated DESC (newest signal at top)
    def _sig_sort_key(s: Dict[str, Any]) -> float:
        ts = s.get("timestamp_generated") or s.get("signal_timestamp") or s.get("timestamp") or ""
        try:
            if "AM" in str(ts) or "PM" in str(ts):
                parsed = datetime.strptime(str(ts), "%d-%b-%Y %I:%M %p")
                return parsed.timestamp()
            return datetime.fromisoformat(str(ts)).timestamp()
        except:
            return 0.0

    merged.sort(key=_sig_sort_key, reverse=True)

    return JSONResponse({"signals": merged})


@app.get("/api/signals/export")
async def api_signals_export(
    format: str = Query("csv"),
    symbol: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    limit: int = Query(2000),
) -> StreamingResponse:
    signals = _load_stored_signals_history(limit=limit)

    # Merge live signals on top if engine is running.
    live: List[Dict[str, Any]] = []
    manager = _engine_manager
    if manager is not None:
        try:
            raw_signals = manager.get_signals(limit=200) or []
            for s in raw_signals:
                serialized = _serialise_signal_for_api(s)
                if "symbol" in serialized:
                    symbol_code = serialized["symbol"]
                    serialized["display_symbol"] = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)
                live.append(serialized)
        except Exception as e:
            logger.error("api_signals_export live merge error: %s", e)

    signals = _merge_and_dedupe_signals(live, signals)

    if symbol:
        sym = str(symbol).strip().upper()
        def _matches_symbol(s: Dict[str, Any], query: str) -> bool:
            # Check direct symbol matches
            if str(s.get("symbol") or "").upper() == query: return True
            if str(s.get("display_symbol") or "").upper() == query: return True
            # Check resolved display name if display_symbol is missing
            symbol_code = s.get("symbol", "")
            if symbol_code:
                disp = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code).upper()
                if disp == query: return True
            return False

        signals = [s for s in signals if _matches_symbol(s, sym)]

    if timeframe:
        tf = str(timeframe).strip().lower()
        signals = [s for s in signals if str(s.get("signal_timeframe") or s.get("timeframe") or "").strip().lower() == tf]

    # Format signals for export with proper display columns matching the UI
    formatted_signals: List[Dict[str, Any]] = []
    for idx, s in enumerate(signals, start=1):
        # Get display symbol
        symbol_code = s.get("symbol", "")
        display_symbol = s.get("display_symbol") or INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)
        
        # Get timestamp
        ts_gen = s.get("timestamp_generated") or s.get("signal_timestamp") or ""
        
        # Get direction
        direction = str(s.get("final_action") or s.get("action") or "").upper()
        
        # Get timeframe
        tf = s.get("signal_timeframe") or s.get("timeframe") or ""
        
        # Get prices
        entry_price = s.get("entry_price")
        sl = s.get("stop_loss_level")
        target = s.get("target_level")
        
        # Get reason
        reason = s.get("strategy_name") or s.get("reason") or ""
        
        formatted_signals.append({
            "Signal ID (Sr. No.)": idx,
            "Generation Time (IST)": ts_gen,
            "Symbol": display_symbol,
            "Direction (Buy/Sell)": direction,
            "Timeframe": tf,
            "Entry Price": float(entry_price) if entry_price is not None else None,
            "Stoploss": float(sl) if sl is not None else None,
            "Target Price": float(target) if target is not None else None,
            "Reason for Trade": reason,
        })

    # Sort by generation time DESC (newest first) - same as UI
    def _sig_sort_key(s: Dict[str, Any]) -> float:
        ts = s.get("Generation Time (IST)") or ""
        try:
            if "AM" in str(ts) or "PM" in str(ts):
                parsed = datetime.strptime(str(ts), "%d-%b-%Y %I:%M %p")
                return parsed.timestamp()
            return datetime.fromisoformat(str(ts)).timestamp()
        except:
            return 0.0

    formatted_signals.sort(key=_sig_sort_key, reverse=True)

    fmt = str(format or "csv").lower().strip()
    ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        payload = json.dumps({"signals": formatted_signals}, indent=2, default=str).encode("utf-8")
        filename = f"signals_{ts}.json"
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
        return StreamingResponse(io.BytesIO(payload), media_type="application/json", headers=headers)

    df = pd.DataFrame(formatted_signals)
    if df.empty:
        csv_bytes = b""
    else:
        csv_bytes = df.to_csv(index=False).encode("utf-8")

    filename = f"signals_{ts}.csv"
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv", headers=headers)


@app.get("/api/market-feed")
async def api_market_feed() -> JSONResponse:
    """Return current live market feed for watchlist symbols."""

    mode = _get_mode()

    # Prefer live WebSocket ticks when available; fall back to REST quotes
    # when the WebSocket feed is not connected.
    use_ws = False
    with _ws_lock:
        ws_mgr = _ws_manager
    if ws_mgr is not None:
        try:
            use_ws = ws_mgr.is_connected()
        except Exception:
            use_ws = False

    if not use_ws:
        _refresh_market_feed_live()

    # Convert to sorted list for stable table rendering (may be empty)
    with _market_feed_lock:
        rows = sorted(_market_feed_cache.values(), key=lambda r: r.get("symbol") or "")

    source = "ws" if use_ws else "rest"
    return JSONResponse({"mode": mode, "rows": rows, "source": source})


@app.get("/api/market-feed-commodities")
async def api_market_feed_commodities() -> JSONResponse:
    """Return current live market feed for commodities watchlist symbols."""

    mode = _get_mode()
    logger.info("🔍 COMMODITIES API CALLED - checking commodity feed")

    use_ws = False
    with _ws_lock:
        ws_mgr = _ws_manager
    if ws_mgr is not None:
        try:
            use_ws = ws_mgr.is_connected()
        except Exception:
            use_ws = False

    if use_ws:
        # Use WebSocket data for real-time updates
        with _commodity_feed_lock:
            rows = sorted(_commodity_feed_cache.values(), key=lambda r: r.get("symbol") or "")
        source = "ws"
        logger.info(f"🔌 WebSocket mode - found {len(rows)} commodities")
    else:
        # Fall back to REST quotes when WebSocket is not available
        logger.info("🔄 REST mode - calling commodity refresh")
        _refresh_commodity_feed_live()
        
        with _commodity_feed_lock:
            rows = sorted(_commodity_feed_cache.values(), key=lambda r: r.get("symbol") or "")
        source = "rest"
        logger.info(f"✅ REST mode - found {len(rows)} commodities")

    return JSONResponse({"mode": mode, "rows": rows, "source": source})


@app.websocket("/ws/market-feed")
async def ws_market_feed(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            mode = _get_mode()
            rows = await _get_market_rows_snapshot()
            with _paper_lock:
                paper_version = _paper_trades_version
                paper_total = len(_paper_trades)

            await websocket.send_json(
                {
                    "mode": mode,
                    "rows": rows,
                    "source": "ws_push",
                    "paper_trades_version": paper_version,
                    "paper_trades_total": paper_total,
                }
            )
            # Push watchlist snapshots twice per second for smoother LTP updates
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    except Exception as e:  # pragma: no cover - network heavy
        logger.error("Browser market-feed WebSocket error: %s", e)


@app.websocket("/ws/candles")
async def ws_candles(websocket: WebSocket) -> None:
    """Stream latest live candle for a symbol/timeframe to the browser.

    The browser connects with query parameters, for example:
    ws://host/ws/candles?symbol=RELIANCE&timeframe=5m
    """

    await websocket.accept()

    params = websocket.query_params
    sym = (params.get("symbol") or "").upper()
    tf = params.get("timeframe") or "5m"

    if not sym:
        await websocket.close(code=1008)
        return

    try:
        while True:
            # Use live candles only; historical backfill still comes from
            # the REST-based /api/candles endpoint.
            df = _get_live_candles(sym, tf, 1)
            if not df.empty:
                row = df.iloc[-1]
                ts_iso = row["date"].isoformat()

                candle = {
                    "date": ts_iso,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                await websocket.send_json(
                    {
                        "symbol": sym,
                        "timeframe": tf,
                        "candle": candle,
                    }
                )

            # Push latest candle up to four times per second so the
            # rightmost bar moves almost in real time.
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception as e:  # pragma: no cover - network heavy
        logger.error("Browser candles WebSocket error: %s", e)


@app.get("/api/candles")
async def api_candles(symbol: str, timeframe: str = "5m", count: int = 200) -> JSONResponse:
    """Return historical OHLCV candles for a given symbol/timeframe.

    All data is fetched directly from the broker's history() endpoint; no
    synthetic or simulated candles are produced here.
    """

    sym = (symbol or "").upper()
    tf = timeframe or "5m"
    try:
        cnt = max(1, min(int(count), 2000))
    except Exception:
        cnt = 200

    # Base: historical candles from REST API
    hist_df = _get_historical_candles(sym, tf, cnt)

    # Overlay: live candles built from WebSocket ticks (if available). We
    # request only a small number because they represent the tail of the
    # series and will be merged on timestamp.
    live_df = _get_live_candles(sym, tf, min(cnt, 100))

    if hist_df.empty and live_df.empty:
        return JSONResponse({"symbol": sym, "timeframe": tf, "candles": []})

    if hist_df.empty:
        combined = live_df
    elif live_df.empty:
        combined = hist_df
    else:
        combined = pd.concat([hist_df, live_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    combined = combined.tail(cnt)

    candles: List[Dict[str, Any]] = []
    for _, row in combined.iterrows():
        candles.append(
            {
                "date": row["date"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    return JSONResponse({"symbol": sym, "timeframe": tf, "candles": candles})


@app.get("/api/positions")
async def api_positions() -> JSONResponse:
    """Return active positions from Fyers or paper trading cache."""

    mode = _get_mode()

    # Positions are always read from broker API when available; for
    # paper trading the UI can treat these as reference only.
    _refresh_positions_live()

    return JSONResponse({"mode": mode, "positions": _positions_cache})


@app.get("/api/orders")
async def api_orders() -> JSONResponse:
    """Return recent order history from Fyers or paper trading cache."""

    mode = _get_mode()

    _refresh_orders_live()

    return JSONResponse({"mode": mode, "orders": _orders_cache})


@app.get("/api/margin")
async def api_margin() -> JSONResponse:
    """Return simple margin / funds snapshot."""

    mode = _get_mode()

    _refresh_margin_live()

    return JSONResponse({"mode": mode, "margin": _margin_cache})


@app.get("/api/paper/status")
async def api_paper_status() -> JSONResponse:
    mode = _get_mode()

    with _paper_lock:
        positions = list(_paper_positions.values())
        live_trades = list(_paper_trades)

    # Merge live in-memory trades with persisted stored trades so that
    # the P&L summary stays in sync with the Trade Log History panel
    # (which also uses stored trades via _load_stored_paper_trades_history).
    stored_trades = _load_stored_paper_trades_history(limit=5000)
    trades = _merge_and_dedupe_trades(live_trades, stored_trades)

    # Count only CLOSED trades for "Total Trades" display
    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
    open_positions = [p for p in positions if p.get("status") == "OPEN"]

    # Compute realized P&L from all CLOSED trades.
    # Accept both "pnl" and "pnl_points" keys (the Trade Log uses pnl_points).
    gross_realized_pnl = 0.0
    total_realized_brokerage = 0.0
    total_realized_statutory = 0.0
    
    for t in closed_trades:
        # Try pnl first, then fall back to pnl_points
        pnl_val = t.get("pnl") if t.get("pnl") is not None else t.get("pnl_points")
        if pnl_val is None:
            # Last fallback: compute from entry/exit prices directly
            try:
                entry = float(t.get("entry_price") or 0.0)
                exit_p = float(t.get("exit_price") or 0.0)
                direction = str(t.get("direction") or "BUY").upper()
                sym_code = str(t.get("symbol") or "")
                qty_v = int(t.get("qty") or 1)
                ls_v = int(t.get("lot_size") or _commodity_lot_size(sym_code))
                points = (exit_p - entry) if direction == "BUY" else (entry - exit_p)
                pnl_val = points * ls_v * qty_v
            except Exception:
                continue
        try:
            val = float(pnl_val)
            gross_realized_pnl += val

            entry_price = float(t.get("entry_price") or 0.0)
            exit_price = float(t.get("exit_price") or 0.0)
            symbol_code = str(t.get("symbol") or "")
            qty = int(t.get("qty") or 1)
            lot_size = int(t.get("lot_size") or _commodity_lot_size(symbol_code))

            turnover = (entry_price + exit_price) * lot_size * qty
            brokerage_info = _calculate_trade_brokerage(symbol_code, entry_price, exit_price, qty)
            total_realized_brokerage += brokerage_info["brokerage"]
            total_realized_statutory += brokerage_info["statutory_charges"]

        except Exception:
            continue

    total_realized_charges = total_realized_brokerage + total_realized_statutory
    net_realized_pnl = gross_realized_pnl - total_realized_charges

    # Unrealized P&L calculation for open positions
    unrealized_gross_pnl = 0.0
    unrealized_charges = 0.0
    
    for p in open_positions:
        pnl_val = p.get("pnl")
        if pnl_val is not None:
            try:
                unrealized_gross_pnl += float(pnl_val)
                continue
            except Exception:
                pass

        symbol_code = str(p.get("symbol") or "")
        entry_price = p.get("entry_price")
        qty = p.get("qty", 1)
        if not symbol_code or entry_price is None:
            continue
        try:
            entry = float(entry_price)
            qty_val = int(qty)
        except Exception:
            continue
        ltp = _get_latest_ltp(symbol_code)
        if ltp is None:
            continue
        direction = str(p.get("direction") or "BUY").upper()
        if direction == "BUY":
            points = ltp - entry
        else:
            points = entry - ltp

        lot_size = p.get("lot_size")
        try:
            lot_size_val = int(lot_size) if lot_size is not None else _commodity_lot_size(symbol_code)
        except Exception:
            lot_size_val = _commodity_lot_size(symbol_code)

        # Unrealized Gross P&L
        gross_unrealized = points * float(lot_size_val) * float(qty_val)
        unrealized_gross_pnl += gross_unrealized
        
        # Estimate charges for unrealized (Entry + Current LTP)
        turnover_unrealized = (entry + ltp) * float(lot_size_val) * float(qty_val)
        est_brokerage_info = _calculate_trade_brokerage(symbol_code, entry, ltp, qty_val)
        unrealized_charges += est_brokerage_info["total_charges"]

    net_unrealized_pnl = unrealized_gross_pnl - unrealized_charges

    # Total P&L calculations
    total_gross_pnl = gross_realized_pnl + unrealized_gross_pnl
    total_net_pnl = net_realized_pnl + net_unrealized_pnl
    
    pnl_breakdown = (
        f"Gross Realized: {gross_realized_pnl:.2f} | "
        f"Brokerage: {total_realized_charges:.2f} | "
        f"Net Realized: {net_realized_pnl:.2f}"
    )

    return JSONResponse(
        {
            "mode": mode,
            "paper_engine_running": _paper_engine_running,
            "engine_running": bool(_engine_manager is not None),
            "open_positions": open_positions,
            "total_trades": len(closed_trades),  # Only count CLOSED trades
            "closed_trades_count": len(closed_trades),
            "open_trades_count": len(open_positions),
            "gross_realized_pnl": round(gross_realized_pnl, 2),
            "total_brokerage": round(total_realized_charges, 2),
            "brokerage_breakdown": {
                "brokerage": round(total_realized_brokerage, 2),
                "statutory_charges": round(total_realized_statutory, 2),
            },
            "net_realized_pnl": round(net_realized_pnl, 2),
            "unrealized_gross_pnl": round(unrealized_gross_pnl, 2),
            "unrealized_charges": round(unrealized_charges, 2),
            "net_unrealized_pnl": round(net_unrealized_pnl, 2),
            "total_gross_pnl": round(total_gross_pnl, 2),
            "total_net_pnl": round(total_net_pnl, 2),
            "realized_pnl": round(net_realized_pnl, 2),  # Backward compatibility
            "unrealized_pnl": round(net_unrealized_pnl, 2),  # Backward compatibility
            "total_pnl": round(total_net_pnl, 2),
            "net_pnl": round(total_net_pnl, 2),
            "is_simulated": mode == "paper",
            "pnl_label": "Net P&L (Excl. Brokerage)" if mode == "paper" else "Real Net P&L (Excl. Brokerage)",
            "pnl_breakdown": pnl_breakdown,
        }
    )


@app.get("/api/paper/positions")
async def api_paper_positions() -> JSONResponse:
    with _paper_lock:
        positions = list(_paper_positions.values())
    return JSONResponse({"positions": positions})


@app.get("/api/paper/trades")
async def api_paper_trades() -> JSONResponse:
    stored = _load_stored_paper_trades_history(limit=5000)
    with _paper_lock:
        live = list(_paper_trades)

    trades = _merge_and_dedupe_trades(live, stored)
    
    # Sort trades by entry_time in ASCENDING order (oldest first) for proper Sr. No. assignment
    # This ensures Sr. No. 1 is the first trade taken, and the sequence is maintained
    def _ts_key_asc(t: Dict[str, Any]) -> str:
        return str(t.get("entry_time") or "")
    
    trades.sort(key=_ts_key_asc, reverse=False)

    # Create display-ready copies without mutating original trade data
    display_trades: List[Dict[str, Any]] = []
    for idx, trade in enumerate(trades, start=1):
        # Create a shallow copy for display purposes only
        display_trade = dict(trade)
        
        # Assign persistent Sr. No. based on chronological order
        display_trade["id"] = idx
        display_trade["Sr. No"] = idx
        
        # Use display_symbol mapping
        if "symbol" in display_trade:
            display_trade["display_symbol"] = INSTRUMENT_TO_DISPLAY.get(display_trade["symbol"], display_trade["symbol"])

        # Map to the new schema requested by the user
        display_trade["Symbol"] = display_trade.get("display_symbol") or display_trade.get("symbol")
        display_trade["Direction"] = str(display_trade.get("direction") or "").upper()
        display_trade["Timeframe"] = display_trade.get("timeframe") or ""
        
        entry_price = display_trade.get("entry_price")
        exit_price = display_trade.get("exit_price")
        
        try:
            display_trade["Entry Price"] = float(entry_price) if entry_price is not None else 0.0
        except:
            display_trade["Entry Price"] = 0.0
            
        try:
            display_trade["Exit Price"] = float(exit_price) if exit_price is not None else None
        except:
            display_trade["Exit Price"] = None

        display_trade["Entry Time (IST)"] = _to_ist_iso(display_trade.get("entry_time")) if display_trade.get("entry_time") else ""
        display_trade["Exit Time (IST)"] = _to_ist_iso(display_trade.get("exit_time")) if display_trade.get("exit_time") else ""
        
        # Points Captured
        pnl_points = float(display_trade.get("pnl_points") or 0.0)
        display_trade["Points Captured"] = pnl_points
        display_trade["Strategy"] = display_trade.get("strategy") or "Aggregated"
        display_trade["Reason"] = str(display_trade.get("reason") or display_trade.get("strategy") or "EngineSignal")
        
        # Calculate individual brokerage for this trade
        if display_trade.get("status") == "CLOSED" and display_trade.get("entry_price") and display_trade.get("exit_price"):
            try:
                brokerage_info = _calculate_trade_brokerage(
                    display_trade.get("symbol", ""),
                    float(display_trade.get("entry_price", 0)),
                    float(display_trade.get("exit_price", 0)),
                    int(display_trade.get("qty", 1))
                )
                display_trade["Brokerage"] = round(brokerage_info["total_charges"], 2)
                display_trade["Brokerage Breakdown"] = {
                    "turnover": round(brokerage_info["total_turnover"], 2),
                    "brokerage": round(brokerage_info["brokerage"], 2),
                    "statutory": round(brokerage_info["statutory_charges"], 2),
                }
                # P&L excluding brokerage
                display_trade["P&L Excl. Brokerage"] = round(pnl_points - brokerage_info["total_charges"], 2)
            except:
                display_trade["Brokerage"] = 0.0
                display_trade["P&L Excl. Brokerage"] = round(pnl_points, 2)
        else:
            display_trade["Brokerage"] = 0.0
            display_trade["P&L Excl. Brokerage"] = round(pnl_points, 2)
        
        # Outcome logic
        close_reason = str(display_trade.get("close_reason") or display_trade.get("reason") or "").lower()
        if "target" in close_reason:
            display_trade["Outcome"] = "TGT Hit"
        elif "stop" in close_reason or "sl" in close_reason:
            display_trade["Outcome"] = "SL Hit"
        else:
            display_trade["Outcome"] = display_trade.get("outcome") or ""
            
        display_trades.append(display_trade)

    # Re-sort for UI display - newest first (but Sr. No. stays consistent)
    def _ts_key_desc(t: Dict[str, Any]) -> str:
        return str(t.get("exit_time") or t.get("entry_time") or "")
    display_trades.sort(key=_ts_key_desc, reverse=True)

    return JSONResponse({"trades": display_trades})


@app.get("/api/trades/export")
async def api_trades_export(
    format: str = Query("csv"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(5000),
) -> StreamingResponse:
    stored = _load_stored_paper_trades_history(limit=limit)
    with _paper_lock:
        live = list(_paper_trades)
    
    trades = _merge_and_dedupe_trades(live, stored)
    
    if symbol:
        sym = str(symbol).strip().upper()
        def _matches_trade_symbol(t: Dict[str, Any], query: str) -> bool:
            # Check direct symbol matches
            if str(t.get("symbol") or "").upper() == query: return True
            if str(t.get("instrument") or "").upper() == query: return True
            if str(t.get("display_symbol") or "").upper() == query: return True
            # Check resolved display name if display_symbol is missing
            symbol_code = t.get("instrument") or t.get("symbol") or ""
            if symbol_code:
                disp = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code).upper()
                if disp == query: return True
            return False

        trades = [t for t in trades if _matches_trade_symbol(t, sym)]

    # If no trades found, return empty CSV with headers only
    if not trades:
        empty_df = pd.DataFrame(columns=[
            "Sr. No", "Symbol", "Direction", "Timeframe", "Strategy", "Entry Price", "Entry Time (IST)",
            "Exit Price", "Exit Time (IST)", "Points", "Brokerage", "P&L (Excl. Brokerage)",
            "Reason", "Outcome"
        ])
        csv_bytes = empty_df.to_csv(index=False).encode("utf-8")
        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        filename = f"trades_{ts}.csv"
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
        return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv", headers=headers)

    # Sort trades by entry_time in ASCENDING order for proper Sr. No. assignment
    def _ts_key_asc(t: Dict[str, Any]) -> str:
        return str(t.get("entry_time") or "")
    trades.sort(key=_ts_key_asc, reverse=False)

    # Format trades for export with proper display columns
    formatted_trades: List[Dict[str, Any]] = []
    for idx, trade in enumerate(trades, start=1):
        # Get display symbol
        symbol_code = trade.get("symbol", "")
        display_symbol = INSTRUMENT_TO_DISPLAY.get(symbol_code, symbol_code)
        
        # Format times
        entry_time = trade.get("entry_time", "")
        exit_time = trade.get("exit_time", "")
        
        # Get prices and P&L
        entry_price = trade.get("entry_price")
        exit_price = trade.get("exit_price")
        pnl_points = float(trade.get("pnl_points") or 0.0)
        
        # Calculate brokerage
        brokerage = 0.0
        pnl_excl_brokerage = pnl_points
        if trade.get("status") == "CLOSED" and entry_price and exit_price:
            try:
                brokerage_info = _calculate_trade_brokerage(
                    symbol_code,
                    float(entry_price),
                    float(exit_price),
                    int(trade.get("qty", 1))
                )
                brokerage = round(brokerage_info["total_charges"], 2)
                pnl_excl_brokerage = round(pnl_points - brokerage, 2)
            except:
                pass
        
        # Determine outcome
        close_reason = str(trade.get("close_reason") or trade.get("reason") or "").lower()
        if "target" in close_reason:
            outcome = "TGT Hit"
        elif "stop" in close_reason or "sl" in close_reason:
            outcome = "SL Hit"
        else:
            outcome = trade.get("outcome") or ""
        
        formatted_trades.append({
            "Sr. No": idx,
            "Symbol": display_symbol,
            "Direction": str(trade.get("direction") or "").upper(),
            "Timeframe": trade.get("timeframe") or "",
            "Strategy": trade.get("strategy") or "Aggregated",
            "Entry Price": float(entry_price) if entry_price is not None else 0.0,
            "Entry Time (IST)": entry_time,
            "Exit Price": float(exit_price) if exit_price is not None else None,
            "Exit Time (IST)": exit_time,
            "Points": round(pnl_points, 2),
            "Brokerage": brokerage,
            "P&L (Excl. Brokerage)": pnl_excl_brokerage,
            "Reason": str(trade.get("reason") or trade.get("strategy") or "EngineSignal"),
            "Outcome": outcome,
        })

    # Re-sort for export - newest first (highest Sr. No. at top)
    formatted_trades.sort(key=lambda x: x["Sr. No"], reverse=True)
    
    logger.info(f"Export trades: {len(formatted_trades)} formatted for export")

    fmt = str(format or "csv").lower().strip()
    ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        payload = json.dumps({"trades": formatted_trades}, indent=2, default=str).encode("utf-8")
        filename = f"trades_{ts}.json"
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
        return StreamingResponse(io.BytesIO(payload), media_type="application/json", headers=headers)

    df = pd.DataFrame(formatted_trades)
    csv_bytes = b"" if df.empty else df.to_csv(index=False).encode("utf-8")
    filename = f"trades_{ts}.csv"
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv", headers=headers)


@app.get("/api/paper/metrics")
async def api_paper_metrics() -> JSONResponse:
    """Real-time paper trading metrics for the dashboard KPI panel.

    Designed to be polled every 2–3 seconds by the frontend without a full
    page refresh. Returns all metrics needed to render the Signal Panel:
    total trades, win/loss ratio, cumulative PnL, active positions, and
    the current market session state (JOINT / MCX_ONLY / CLOSED).
    """
    with _paper_lock:
        trades = list(_paper_trades)
        positions = list(_paper_positions.values())

    # --- Merge with stored history so metrics survive restarts ----------
    stored = _load_stored_paper_trades_history(limit=5000)
    all_trades = _merge_and_dedupe_trades(trades, stored)

    # Only count CLOSED trades for "Total Trades" display
    closed = [t for t in all_trades if t.get("status") == "CLOSED"]
    wins   = [t for t in closed if float(t.get("pnl") or t.get("pnl_points") or 0.0) > 0]
    losses = [t for t in closed if float(t.get("pnl") or t.get("pnl_points") or 0.0) <= 0]

    total_trades  = len(closed)  # Only closed trades count
    win_count     = len(wins)
    loss_count    = len(losses)
    win_rate_pct  = round((win_count / total_trades * 100.0), 1) if total_trades else 0.0
    
    # Calculate cumulative P&L excluding brokerage
    cumulative_pnl = 0.0
    total_brokerage = 0.0
    for t in closed:
        pnl_val = float(t.get("pnl") or t.get("pnl_points") or 0.0)
        cumulative_pnl += pnl_val
        
        # Calculate brokerage for this trade
        try:
            entry_price = float(t.get("entry_price") or 0.0)
            exit_price = float(t.get("exit_price") or 0.0)
            symbol_code = str(t.get("symbol") or "")
            qty = int(t.get("qty") or 1)
            brokerage_info = _calculate_trade_brokerage(symbol_code, entry_price, exit_price, qty)
            total_brokerage += brokerage_info["total_charges"]
        except:
            pass
    
    # Net P&L (excluding brokerage)
    net_cumulative_pnl = cumulative_pnl - total_brokerage
    
    active_positions = len([p for p in positions if p.get("status") == "OPEN"])

    timer = get_market_timer_status()

    # --- Build unified trade log entries --------------------------------
    trade_log: List[Dict[str, Any]] = []
    for i, t in enumerate(sorted(all_trades, key=lambda x: str(x.get("entry_time") or ""), reverse=True), 1):
        status = t.get("status", "OPEN")
        exit_time_raw = t.get("exit_time")
        exit_time_str = exit_time_raw if exit_time_raw else "LIVE / OPEN"
        exit_price_raw = t.get("exit_price")
        exit_price_str = f"({exit_price_raw})" if status == "OPEN" and exit_price_raw else (str(exit_price_raw) if exit_price_raw else "")
        pnl_val = float(t.get("pnl") or t.get("pnl_points") or 0.0)

        trade_log.append({
            "Trade_ID": f"T-{i:04d}",
            "Status": status,
            "Symbol": t.get("display_symbol") or t.get("symbol") or "",
            "Direction": str(t.get("direction") or "").upper(),
            "Timeframe": t.get("timeframe") or "",
            "Entry_Price": t.get("entry_price") or 0.0,
            "Exit_Price": exit_price_str,
            "Entry_Time": t.get("entry_time") or "",
            "Exit_Time": exit_time_str,
            "Stoploss": t.get("stop_loss") or 0.0,
            "Target": t.get("target") or 0.0,
            "Points_Captured": round(float(t.get("pnl_points") or 0.0), 2),
            "Net_PnL": round(pnl_val, 2),
            "Outcome": t.get("outcome") or (
                "TGT Hit" if "Target" in str(t.get("exit_reason") or "") else
                "SL Hit"  if "Stop"   in str(t.get("exit_reason") or "") else ""
            ),
            "Reason": t.get("reason") or t.get("strategy") or "EngineSignal",
        })

    ws_is_live = _is_ws_live()
    ws_tick_age = round(time.time() - _last_tick_time, 1) if _last_tick_time > 0 else -1

    return JSONResponse({
        "Cumulative_PnL":    round(net_cumulative_pnl, 2),  # Net P&L excluding brokerage
        "Gross_Cumulative_PnL": round(cumulative_pnl, 2),   # Gross P&L before charges
        "Total_Brokerage":   round(total_brokerage, 2),     # Total brokerage paid
        "Total_Trades":      total_trades,
        "Win_Rate":          f"{win_rate_pct}%",
        "Active_Positions":  active_positions,
        "Win_Loss_Ratio":    f"{win_count} / {loss_count}",
        "Market_Session":    timer["session"],
        "Equity_Active":     timer["EQUITY_ACTIVE"],
        "MCX_Active":        timer["MCX_ACTIVE"],
        "Current_Time_IST":  timer["current_time_ist"],
        "WS_Live":           ws_is_live,
        "WS_Tick_Age_S":     ws_tick_age,
        "Trade_Log":         trade_log,
    })




@app.get("/api/market/timer")
async def api_market_timer() -> JSONResponse:
    """Returns the current Dual-Market session state in real-time IST."""
    return JSONResponse(get_market_timer_status())


@app.get("/api/health")
async def api_health() -> JSONResponse:
    """Simple health check endpoint for monitoring."""

    status: Dict[str, Any] = {
        "status": "ok",
        "mode": _get_mode(),
        "engine_running": False,
        "market_open_india": _is_market_open_india(),
        "equity_market_open": _is_equity_market_open(),
        "commodity_market_open": _is_commodity_market_open(),
        "ws_connected": False,
    }

    if _engine_manager is not None:
        try:
            engine_status = _engine_manager.get_status()
            # get_status() may return complex objects (e.g. datetimes) inside
            # its dict, which are not JSON-serializable. We only expose a
            # minimal, safe view here.
            if isinstance(engine_status, dict):
                is_running = engine_status.get("is_running")
                status["engine_running"] = bool(is_running)

                # Provide a human-readable summary without embedding the
                # entire nested structure.
                if "status" in engine_status and isinstance(engine_status["status"], str):
                    status["engine_status"] = engine_status["status"]
                else:
                    status["engine_status"] = "running" if status["engine_running"] else "not_running"
            else:
                # Fallback: just stringify whatever was returned.
                status["engine_status"] = str(engine_status)
        except Exception as e:
            status["engine_error"] = str(e)

    with _ws_lock:
        ws_mgr = _ws_manager
    if ws_mgr is not None:
        try:
            status["ws_connected"] = bool(ws_mgr.is_connected())
        except Exception as e:
            status["ws_error"] = str(e)

    # Expose the strict WS liveness check used by the paper engine
    status["ws_live"] = _is_ws_live()
    status["ws_tick_age_s"] = round(time.time() - _last_tick_time, 1) if _last_tick_time > 0 else -1
    status["ws_tick_max_age_s"] = _WS_TICK_MAX_AGE_S

    return JSONResponse(status)


@app.get("/api/health/feed")
async def api_health_feed() -> JSONResponse:
    """System health endpoint for the dashboard health panel.

    Returns live info about the WebSocket tick feed:
      - last_tick_at:          IST timestamp of the most recent tick
      - last_tick_age_seconds: seconds since last tick (-1 if never received)
      - connection_status:     'Online' | 'Offline'
      - total_ticks_today:     cumulative tick count since midnight IST
      - queue_depth:           ticks waiting in the processor queue
      - market_open:           True if any market session is active
    """
    now = time.time()
    with _tick_count_lock:
        last_t = _last_tick_time
        ticks_today = _total_ticks_today

    age = round(now - last_t, 1) if last_t > 0 else -1

    if last_t > 0:
        last_tick_ist = datetime.fromtimestamp(last_t, IST).strftime("%Y-%m-%dT%H:%M:%S%z")
    else:
        last_tick_ist = "never"

    with _ws_lock:
        mgr = _ws_manager
    # Consider online if last tick was within 90 s (handles brief lags)
    connected = bool(mgr is not None and mgr.is_connected()) or (last_t > 0 and (now - last_t) < 90)

    market_open = _is_commodity_market_open() or _is_equity_market_open()
    q_depth = _tick_queue.qsize()

    return JSONResponse(
        {
            "last_tick_at": last_tick_ist,
            "last_tick_age_seconds": age,
            "connection_status": "Online" if connected else "Offline",
            "total_ticks_today": ticks_today,
            "queue_depth": q_depth,
            "market_open": market_open,
            "watchdog_timeout_s": _WS_WATCHDOG_TIMEOUT_S,
        }
    )


@app.get("/api/debug/trades")
async def api_debug_trades() -> JSONResponse:
    """Debug endpoint to check trade counts and data availability."""
    stored = _load_stored_paper_trades_history(limit=5000)
    with _paper_lock:
        live = list(_paper_trades)
    
    trades = _merge_and_dedupe_trades(live, stored)
    
    # Calculate PnL breakdown
    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
    pnl_breakdown = []
    total_pnl = 0.0
    for t in closed_trades:
        pnl = float(t.get("pnl") or t.get("pnl_points") or 0.0)
        total_pnl += pnl
        pnl_breakdown.append({
            "trade_id": t.get("trade_id", "N/A"),
            "symbol": t.get("symbol", "N/A"),
            "entry_time": t.get("entry_time", "N/A"),
            "pnl": pnl,
        })
    
    return JSONResponse({
        "stored_count": len(stored),
        "live_count": len(live),
        "after_merge_count": len(trades),
        "closed_trades_count": len(closed_trades),
        "total_pnl": round(total_pnl, 2),
        "pnl_breakdown": pnl_breakdown,
        "trade_ids_in_stored": [t.get("trade_id") for t in stored[:10]],
        "trade_ids_in_live": [t.get("trade_id") for t in live[:10]],
    })


@app.get("/api/trades/export-raw")
async def api_trades_export_raw() -> StreamingResponse:
    """Direct raw export that bypasses all processing - for debugging."""
    # Read the raw JSON file directly
    base_dir = _paper_trades_storage_dir()
    today = date.today()
    fp = os.path.join(base_dir, today.strftime("%Y-%m-%d"), f"paper_trades_{today.strftime('%Y-%m-%d')}.json")
    
    raw_data = []
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    raw_data = json.loads(content)
        except Exception as e:
            return StreamingResponse(
                io.BytesIO(f"Error reading file: {e}".encode()),
                media_type="text/plain"
            )
    
    # Return as JSON download
    payload = json.dumps({"trades": raw_data}, indent=2, default=str).encode("utf-8")
    headers = {"Content-Disposition": f"attachment; filename=\"raw_trades_{today}.json\""}
    return StreamingResponse(io.BytesIO(payload), media_type="application/json", headers=headers)


