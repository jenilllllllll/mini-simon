from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from fyers_apiv3.FyersWebsocket import data_ws
from fyers_apiv3 import fyersModel

from config import get_config
from live_engine import EngineManager


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
    "JIOFINANCE": "NSE:JIOFINANCE-EQ",
}


price_buffer_lock = threading.Lock()
price_buffer: Dict[str, Dict[str, Any]] = {}

candle_buffer_lock = threading.Lock()
candle_buffer: Dict[str, List[Dict[str, Any]]] = {}

trade_log_lock = threading.Lock()
trade_log: List[Dict[str, Any]] = []

positions_lock = threading.Lock()
open_positions: Dict[str, Dict[str, Any]] = {}

fyers_rest_client_lock = threading.Lock()
_fyers_rest_client: Optional[fyersModel.FyersModel] = None

ws_manager_lock = threading.Lock()
_ws_manager: Optional["WebSocketManager"] = None


engine_manager_lock = threading.Lock()
_engine_manager: Optional[EngineManager] = None


st.set_page_config(page_title="Nifty-50 Algorithmic Trading Terminal", layout="wide")


def _init_session_state() -> None:
    """Initialize session_state keys for symbol selection and paper trading."""

    if "selected_symbol" not in st.session_state:
        st.session_state["selected_symbol"] = sorted(NIFTY_50_SYMBOLS.keys())[0]

    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []

    if "paper_positions" not in st.session_state:
        st.session_state["paper_positions"] = {}

    if "paper_last_candle" not in st.session_state:
        # Maps symbol_code -> last processed candle timestamp
        st.session_state["paper_last_candle"] = {}


def _apply_dark_theme() -> None:
    """Inject a dark, high-density theme and card styling via CSS."""

    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0%, #05070a 55%, #020308 100%);
            color: #e5e5e5;
        }

        .block-container {
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
            max-width: 100% !important;
        }

        /* Center top-level tabs */
        div[data-baseweb="tab-list"] {
            justify-content: center !important;
        }

        .card {
            background-color: #14141c;
            border: 1px solid #333;
            border-radius: 0.6rem;
            padding: 0.6rem 0.8rem;
            box-shadow: 0 0 12px rgba(0, 0, 0, 0.6);
        }

        .exec-card .stButton>button {
            border-radius: 4px;
            font-weight: 600;
            border: none;
            padding: 0.4rem 0.6rem;
        }

        .exec-card div.stButton:nth-of-type(1) button {
            background-color: #22c55e;
            color: #0b0b0f;
        }

        .exec-card div.stButton:nth-of-type(2) button {
            background-color: #f97373;
            color: #0b0b0f;
        }

        .kpi-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.75;
        }

        .kpi-value {
            font-size: 1.4rem;
            font-weight: 600;
        }

        .kpi-green {
            color: #22c55e;
        }

        .kpi-red {
            color: #f97373;
        }

        .kpi-neutral {
            color: #e5e5e5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_fyers_auth() -> Tuple[str, str]:
    """Return (app_id, access_token) using config.data_feed.* settings.

    The access token is formatted as required by Fyers: "appid:accesstoken".
    """

    cfg = get_config()
    app_id = cfg.get("data_feed.app_id")
    access_token_value = cfg.get("data_feed.access_token")
    if not app_id or not access_token_value:
        raise RuntimeError("Fyers API credentials are not configured in config.data_feed")

    access_token_str = str(access_token_value)
    if ":" in access_token_str:
        token = access_token_str
    else:
        token = f"{app_id}:{access_token_str}"
    return app_id, token


def get_fyers_rest() -> fyersModel.FyersModel:
    """Singleton REST client for history() calls used in backtesting."""

    global _fyers_rest_client
    if _fyers_rest_client is not None:
        return _fyers_rest_client

    app_id, token = get_fyers_auth()
    with fyers_rest_client_lock:
        if _fyers_rest_client is None:
            _fyers_rest_client = fyersModel.FyersModel(
                client_id=app_id,
                token=token,
                log_path="",
            )
    return _fyers_rest_client


@st.cache_resource
def _init_engine_manager() -> Optional[EngineManager]:
    """Initialize and start the LiveEngine via EngineManager.

    Uses the same FYERS credentials and symbol list as the dashboard config,
    so the engine runs the real strategies (vol spike, body imbalance, etc.)
    on live data.
    """

    try:
        app_id, token = get_fyers_auth()
    except Exception as e:  # pragma: no cover - defensive
        print("EngineManager init error while reading FYERS auth:", e)
        return None

    try:
        manager = EngineManager()

        # Patch the data_feed section of the EngineManager config
        cfg = get_config()
        data_cfg = manager.config.get("data_feed", {})
        data_cfg["app_id"] = app_id
        data_cfg["access_token"] = token

        symbols = cfg.get("data_feed.symbols")
        if symbols:
            data_cfg["symbols"] = symbols

        timeframes = cfg.get("data_feed.timeframes")
        if timeframes:
            data_cfg["timeframes"] = timeframes

        manager.config["data_feed"] = data_cfg

        started = manager.start_engine()
        if not started:
            print("Failed to start LiveEngine from dashboard")
            return None

        return manager

    except Exception as e:  # pragma: no cover - defensive
        print("EngineManager init error:", e)
        return None


def get_engine_manager() -> Optional[EngineManager]:
    """Return cached EngineManager instance (or None if startup failed)."""

    return _init_engine_manager()


def _update_price_and_candle(symbol: str, ltp: float, volume: float, ts: Optional[datetime] = None) -> None:
    """Update shared price buffer and build 1m candles from real WebSocket ticks."""

    if ts is None:
        ts = datetime.utcnow()

    with price_buffer_lock:
        price_buffer[symbol] = {"ltp": float(ltp), "timestamp": ts, "volume": float(volume)}

    minute_ts = ts.replace(second=0, microsecond=0)
    with candle_buffer_lock:
        candles = candle_buffer.get(symbol)
        if candles is None:
            candles = []
            candle_buffer[symbol] = candles

        if candles and candles[-1]["time"] == minute_ts:
            c = candles[-1]
            if ltp > c["high"]:
                c["high"] = ltp
            if ltp < c["low"]:
                c["low"] = ltp
            c["close"] = ltp
            c["volume"] = float(c["volume"]) + float(volume)
        else:
            candles.append(
                {
                    "time": minute_ts,
                    "open": ltp,
                    "high": ltp,
                    "low": ltp,
                    "close": ltp,
                    "volume": float(volume),
                }
            )

        max_len = 500
        if len(candles) > max_len:
            del candles[0 : len(candles) - max_len]


def get_price_snapshot() -> Dict[str, Dict[str, Any]]:
    """Thread-safe shallow copy of current price buffer."""

    with price_buffer_lock:
        return {k: dict(v) for k, v in price_buffer.items()}


def get_candles_snapshot(symbol: str, limit: int = 200) -> pd.DataFrame:
    """Return recent 1m candles for a symbol as a DataFrame."""

    with candle_buffer_lock:
        candles = candle_buffer.get(symbol, [])[-limit:]
        data = [dict(c) for c in candles]

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values("time")
    df["date"] = df["time"]
    return df[["date", "open", "high", "low", "close", "volume"]]


class WebSocketManager:
    """Wrapper around FyersDataSocket (data_ws) using SymbolUpdate mode.

    Runs in the background (Fyers internal threads) and feeds the shared price
    and candle buffers.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.fyers_ws: Optional[data_ws.FyersDataSocket] = None
        self._connected = threading.Event()

    def onmessage(self, message: Dict[str, Any]) -> None:
        try:
            if not isinstance(message, dict):
                return

            symbol = message.get("symbol")
            if not symbol or symbol not in self.symbols:
                return

            if "ltp" not in message:
                return

            ltp = float(message["ltp"])
            volume = float(message.get("volume", 0.0))

            ts_val = message.get("timestamp")
            if ts_val is not None:
                try:
                    ts = datetime.fromtimestamp(ts_val)
                except Exception:
                    ts = datetime.utcnow()
            else:
                ts = datetime.utcnow()

            _update_price_and_candle(symbol, ltp, volume, ts)

        except Exception as e:
            print(f"WebSocket onmessage error: {e}")

    def onerror(self, message: Any) -> None:
        print("WebSocket error:", message)

    def onclose(self, message: Any) -> None:
        print("WebSocket closed:", message)
        self._connected.clear()

    def onopen(self) -> None:
        print("WebSocket opened, subscribing to Nifty-50 symbols...")
        try:
            self._connected.set()
            self.fyers_ws.subscribe(
                symbols=self.symbols,
                data_type="SymbolUpdate",
                channel=15,
            )
            self.fyers_ws.keep_running()
        except Exception as e:
            print(f"Error in WebSocket onopen: {e}")

    def start(self) -> None:
        if self.fyers_ws is not None:
            return

        _, token = get_fyers_auth()

        self.fyers_ws = data_ws.FyersDataSocket(
            access_token=token,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self.onopen,
            on_close=self.onclose,
            on_error=self.onerror,
            on_message=self.onmessage,
        )

        t = threading.Thread(target=self.fyers_ws.connect, daemon=True)
        t.start()


def get_ws_manager() -> WebSocketManager:
    global _ws_manager
    with ws_manager_lock:
        if _ws_manager is None:
            symbols = list(NIFTY_50_SYMBOLS.values())
            _ws_manager = WebSocketManager(symbols=symbols)
            _ws_manager.start()
    return _ws_manager


def get_open_positions_snapshot() -> Dict[str, Dict[str, Any]]:
    """Return current paper positions snapshot from session_state."""

    positions = st.session_state.get("paper_positions", {})
    return {k: dict(v) for k, v in positions.items()}


def get_trades_with_pnl_snapshot() -> List[Dict[str, Any]]:
    """Return paper trades with realized / unrealized P&L, using session_state log."""

    prices = get_price_snapshot()
    trades_base: List[Dict[str, Any]] = st.session_state.get("paper_trades", [])
    trades_copy = [dict(t) for t in trades_base]

    enriched: List[Dict[str, Any]] = []
    for t in trades_copy:
        symbol = t.get("symbol")
        direction = t.get("direction", "BUY")
        qty = int(t.get("qty", 1))
        entry_price = float(t.get("entry_price", 0.0))
        status = t.get("status", "OPEN")

        pnl_points = 0.0
        pnl_value = 0.0
        unrealized = 0.0

        if status == "CLOSED" and t.get("exit_price") is not None:
            exit_price = float(t["exit_price"])
            if direction == "BUY":
                pnl_points = exit_price - entry_price
            else:
                pnl_points = entry_price - exit_price
            pnl_value = pnl_points * qty
        else:
            ltp = None
            if symbol in prices:
                ltp_val = prices[symbol].get("ltp")
                if ltp_val is not None:
                    ltp = float(ltp_val)

            if ltp is not None and ltp > 0:
                if direction == "BUY":
                    pnl_points = ltp - entry_price
                else:
                    pnl_points = entry_price - ltp
                pnl_value = pnl_points * qty
                unrealized = pnl_value

        result = dict(t)
        result["points"] = pnl_points
        result["pnl"] = pnl_value
        result["unrealized_pnl"] = unrealized
        enriched.append(result)

    # Persist derived fields back into session_state for consistency
    st.session_state["paper_trades"] = enriched
    return enriched


def compute_pnl_matrix(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_trades = len(trades)
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    wins = [t for t in closed if t.get("pnl", 0.0) > 0]
    losses = [t for t in closed if t.get("pnl", 0.0) < 0]

    win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
    total_pnl = sum(float(t.get("pnl", 0.0)) for t in trades)

    cum_pnl = 0.0
    max_equity = 0.0
    max_drawdown = 0.0

    def _ts(tr: Dict[str, Any]) -> datetime:
        return tr.get("exit_time") or tr.get("entry_time") or datetime.utcnow()

    for t in sorted(trades, key=_ts):
        cum_pnl += float(t.get("pnl", 0.0))
        max_equity = max(max_equity, cum_pnl)
        dd = max_equity - cum_pnl
        if dd > max_drawdown:
            max_drawdown = dd

    return {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
    }


def _ensure_paper_state() -> None:
    """Ensure paper trading state structures exist in session_state."""

    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []
    if "paper_positions" not in st.session_state:
        st.session_state["paper_positions"] = {}
    if "paper_last_candle" not in st.session_state:
        st.session_state["paper_last_candle"] = {}


def _open_paper_trade(
    symbol_code: str,
    display_symbol: str,
    direction: str,
    price: float,
    when: datetime,
    timeframe: str,
    strategy: str,
    reason: str,
) -> None:
    """Open a new simulated trade and store it in session_state."""

    _ensure_paper_state()
    trades: List[Dict[str, Any]] = st.session_state["paper_trades"]
    positions: Dict[str, Dict[str, Any]] = st.session_state["paper_positions"]

    trade = {
        "instrument": display_symbol,
        "symbol": symbol_code,
        "direction": direction,
        "qty": 1,
        "entry_price": float(price),
        "entry_time": when,
        "exit_price": None,
        "exit_time": None,
        "status": "OPEN",
        "timeframe": timeframe,
        "strategy": strategy,
        "reason": reason,
    }

    trades.append(trade)
    positions[symbol_code] = trade
    st.session_state["paper_trades"] = trades
    st.session_state["paper_positions"] = positions


def _close_paper_trade(symbol_code: str, exit_price: float, when: datetime, reason: str) -> None:
    """Close an existing simulated trade for the given symbol code."""

    _ensure_paper_state()
    positions: Dict[str, Dict[str, Any]] = st.session_state["paper_positions"]
    trades: List[Dict[str, Any]] = st.session_state["paper_trades"]

    pos = positions.get(symbol_code)
    if not pos:
        return

    direction = pos.get("direction", "BUY")
    qty = int(pos.get("qty", 1))
    entry_price = float(pos.get("entry_price", 0.0))

    if direction == "BUY":
        pnl_points = float(exit_price) - entry_price
    else:
        pnl_points = entry_price - float(exit_price)
    pnl_value = pnl_points * qty

    # Update the trade dict in place
    pos["exit_price"] = float(exit_price)
    pos["exit_time"] = when
    pos["status"] = "CLOSED"
    pos["points"] = pnl_points
    pos["pnl"] = pnl_value
    pos["close_reason"] = reason

    # Sync back to the list (in case of distinct dicts)
    for t in trades:
        if t is pos:
            break
        if (
            t.get("symbol") == symbol_code
            and t.get("entry_time") == pos.get("entry_time")
            and t.get("status") != "CLOSED"
        ):
            t.update(pos)

    positions.pop(symbol_code, None)
    st.session_state["paper_positions"] = positions
    st.session_state["paper_trades"] = trades


def run_paper_trading_step(selected_symbol: str) -> None:
    """Simple paper-trading loop based on new 1m candles for the selected symbol.

    This is a placeholder strategy using candle body movement as the condition.
    Replace the thresholds/logic here with your production strategy rules.
    """

    _ensure_paper_state()

    symbol_code = NIFTY_50_SYMBOLS[selected_symbol]
    df = get_candles_snapshot(symbol_code, limit=50)
    if df.empty or len(df) < 3:
        return

    last_row = df.iloc[-1]
    last_ts = last_row["date"]
    if isinstance(last_ts, pd.Timestamp):
        last_ts = last_ts.to_pydatetime()

    last_processed = st.session_state["paper_last_candle"].get(symbol_code)
    if isinstance(last_processed, pd.Timestamp):
        last_processed = last_processed.to_pydatetime()

    # Process each candle only once
    if last_processed is not None and last_ts <= last_processed:
        return

    st.session_state["paper_last_candle"][symbol_code] = last_ts

    close_price = float(last_row["close"])
    open_price = float(last_row["open"])
    if close_price <= 0 or open_price <= 0:
        return

    positions: Dict[str, Dict[str, Any]] = st.session_state["paper_positions"]
    pos = positions.get(symbol_code)

    body = close_price - open_price
    body_pct = body / open_price

    # Manage existing position (basic target / stop-loss)
    if pos is not None:
        entry = float(pos.get("entry_price", close_price))
        direction = pos.get("direction", "BUY")
        move = (close_price - entry) / entry if entry > 0 else 0.0
        if direction == "SELL":
            move = -move

        target_pct = 0.01   # +1%
        stop_pct = -0.005   # -0.5%

        if move >= target_pct or move <= stop_pct:
            reason = "Target hit" if move >= target_pct else "Stop loss hit"
            _close_paper_trade(symbol_code, close_price, last_ts, reason)
        return

    # No open position: check for new entry based on candle body
    threshold_pct = 0.003  # 0.3% body move
    if abs(body_pct) < threshold_pct:
        return

    direction = "BUY" if body > 0 else "SELL"
    reason = f"Candle body move {body_pct * 100:.2f}%"

    _open_paper_trade(
        symbol_code=symbol_code,
        display_symbol=selected_symbol,
        direction=direction,
        price=close_price,
        when=last_ts,
        timeframe="1m",
        strategy="CandleBodyBreakout",
        reason=reason,
    )


def fetch_history(symbol: str, timeframe: str, start: date, end: date) -> pd.DataFrame:
    tf_map = {
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

    resolution = tf_map.get(timeframe)
    if resolution is None:
        raise ValueError(f"Unsupported timeframe for backtest: {timeframe}")

    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to": end.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    client = get_fyers_rest()
    resp = client.history(data)

    if resp.get("s") != "ok":
        print("History error:", resp)
        return pd.DataFrame()

    candles = resp.get("candles", [])
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


def run_backtest(symbol_name: str, timeframe: str, start: date, end: date) -> pd.DataFrame:
    """Placeholder backtest that currently does not use the old momentum demo.

    The momentum-based backtest has been removed. This function can be
    extended to run the production strategies (via FeatureEngine,
    LiveStrategyRunner, and SignalAggregator) over historical data.
    """

    _ = (symbol_name, timeframe, start, end)
    return pd.DataFrame()


def render_live_tab() -> None:
    """High-density Live Dashboard with watchlist, chart, log, and signals."""

    get_ws_manager()

    selected_symbol = st.session_state.get("selected_symbol", sorted(NIFTY_50_SYMBOLS.keys())[0])
    symbol_code = NIFTY_50_SYMBOLS[selected_symbol]

    # Advance the paper trading engine based on the latest candle
    run_paper_trading_step(selected_symbol)

    col_left, col_mid, col_right = st.columns([1.5, 4, 1.5])

    # Left: Watchlist
    with col_left:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Watchlist")

            symbols_sorted = sorted(NIFTY_50_SYMBOLS.keys())
            try:
                current_index = symbols_sorted.index(selected_symbol)
            except ValueError:
                current_index = 0

            choice = st.radio(
                "Nifty-50 Stocks",
                options=symbols_sorted,
                index=current_index,
                label_visibility="collapsed",
            )
            if choice != selected_symbol:
                st.session_state["selected_symbol"] = choice
                selected_symbol = choice
                symbol_code = NIFTY_50_SYMBOLS[selected_symbol]

            st.markdown("</div>", unsafe_allow_html=True)

    # Middle: Candlestick chart (top) + Operational Log (bottom)
    with col_mid:
        # Chart card
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### {selected_symbol} – Live Candlestick")

            prices = get_price_snapshot()
            latest = prices.get(symbol_code)
            ltp = float(latest["ltp"]) if latest and latest.get("ltp") is not None else None

            if ltp is not None:
                st.caption(f"LTP: {ltp:.2f}")
            else:
                st.caption("LTP: –")

            df = get_candles_snapshot(symbol_code, limit=180)
            if df.empty:
                st.info("Waiting for live data from Fyers WebSocket (SymbolUpdate)...")
            else:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df["date"],
                            open=df["open"],
                            high=df["high"],
                            low=df["low"],
                            close=df["close"],
                            name=selected_symbol,
                            increasing_line_color="#22c55e",
                            decreasing_line_color="#f97373",
                        )
                    ]
                )
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=420,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Operational log card
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Operational Log")

            trades = get_trades_with_pnl_snapshot()
            symbol_trades = [
                t
                for t in trades
                if t.get("symbol") == symbol_code or t.get("instrument") == selected_symbol
            ]

            if symbol_trades:
                df_log = pd.DataFrame(symbol_trades)
                for col in ["entry_time", "exit_time"]:
                    if col in df_log.columns:
                        df_log[col] = df_log[col].astype(str)

                df_log = df_log.sort_values(by="entry_time", ascending=False)
                st.dataframe(df_log, use_container_width=True, height=220)

                csv = df_log.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Operational Log CSV",
                    data=csv,
                    file_name="operational_log.csv",
                    mime="text/csv",
                    key="operational_log_download",
                )
            else:
                st.info("No simulated trades yet for this instrument.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Right: Live Signals + Execution panel + Position summary
    with col_right:
        # Live signals card
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Live Signals")

            manager = get_engine_manager()
            if manager is None:
                st.warning("Engine is not running. Check FYERS credentials / access token.")
            else:
                engine_signals: List[Dict[str, Any]] = []
                try:
                    engine_signals = manager.get_signals(limit=50) or []
                except Exception as e:
                    st.error(f"Error fetching engine signals: {e}")

                if engine_signals:
                    df_sig = pd.DataFrame(engine_signals)

                    # Try to filter to the selected symbol if symbol column exists
                    symbol_cols = [c for c in df_sig.columns if c.lower() in {"symbol", "instrument"}]
                    if symbol_cols:
                        sym_col = symbol_cols[0]
                        df_sig = df_sig[df_sig[sym_col] == symbol_code]

                    # Show only the most relevant columns if present
                    preferred_cols = [
                        "signal_timestamp",
                        "symbol",
                        "timeframe",
                        "action",
                        "direction",
                        "confidence",
                        "strategy_name",
                    ]
                    cols_to_show = [c for c in preferred_cols if c in df_sig.columns]
                    if cols_to_show:
                        df_sig = df_sig[cols_to_show]

                    for col in df_sig.columns:
                        if "time" in col.lower():
                            df_sig[col] = df_sig[col].astype(str)

                    if df_sig.empty:
                        st.info("No recent engine signals for this instrument.")
                    else:
                        st.dataframe(df_sig, use_container_width=True, height=220)
                else:
                    st.info("No engine signals available yet.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Execution panel card
        with st.container():
            st.markdown('<div class="card exec-card">', unsafe_allow_html=True)
            st.markdown("#### Execution Panel")
            st.caption("Simulated paper orders (no live execution).")

            prices = get_price_snapshot()
            latest = prices.get(symbol_code)
            ltp = float(latest["ltp"]) if latest and latest.get("ltp") is not None else None

            qty_default = int(st.session_state.get("exec_qty", 1) or 1)
            qty = st.number_input(
                "Quantity",
                min_value=1,
                max_value=100000,
                value=qty_default,
                step=1,
                key="exec_qty",
            )

            col_buy, col_sell = st.columns(2)
            with col_buy:
                if st.button("BUY", use_container_width=True, key="exec_buy"):
                    if ltp is not None:
                        _open_paper_trade(
                            symbol_code=symbol_code,
                            display_symbol=selected_symbol,
                            direction="BUY",
                            price=ltp,
                            when=datetime.utcnow(),
                            timeframe="1m",
                            strategy="Manual",
                            reason="Manual BUY",
                        )
            with col_sell:
                if st.button("SELL", use_container_width=True, key="exec_sell"):
                    if ltp is not None:
                        _open_paper_trade(
                            symbol_code=symbol_code,
                            display_symbol=selected_symbol,
                            direction="SELL",
                            price=ltp,
                            when=datetime.utcnow(),
                            timeframe="1m",
                            strategy="Manual",
                            reason="Manual SELL",
                        )

            # Optional quick export of the full paper trade log
            trades_all = st.session_state.get("paper_trades", [])
            if trades_all:
                df_all = pd.DataFrame(trades_all)
                for col in ["entry_time", "exit_time"]:
                    if col in df_all.columns:
                        df_all[col] = df_all[col].astype(str)
                csv_all = df_all.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="SAVE CSV",
                    data=csv_all,
                    file_name="trade_log.csv",
                    mime="text/csv",
                    key="exec_save_csv",
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Current positions card
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Current Positions")

            prices = get_price_snapshot()
            ltp = None
            latest = prices.get(symbol_code)
            if latest and latest.get("ltp") is not None:
                ltp = float(latest["ltp"])

            positions = get_open_positions_snapshot()
            pos = positions.get(symbol_code)

            if not pos:
                st.info("No open paper positions for the selected instrument.")
            else:
                entry = float(pos.get("entry_price", 0.0))
                direction = pos.get("direction", "BUY")
                qty = int(pos.get("qty", 1))

                if ltp is not None and entry > 0:
                    if direction == "BUY":
                        points = ltp - entry
                    else:
                        points = entry - ltp
                    pnl = points * qty
                else:
                    points = 0.0
                    pnl = 0.0

                st.markdown(
                    f"**Direction:** {direction}  |  **Qty:** {qty}  |  **Entry:** {entry:.2f}"
                )
                if ltp is not None:
                    st.markdown(f"**LTP:** {ltp:.2f}")

                pnl_color = "#22c55e" if pnl >= 0 else "#f97373"
                st.markdown(
                    f"Unrealized P&L: <span style='color:{pnl_color}'>{pnl:.2f} INR</span>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


def render_trades_tab() -> None:
    st.subheader("Paper Trade Log & P&L")

    trades = get_trades_with_pnl_snapshot()
    matrix = compute_pnl_matrix(trades)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", matrix["total_trades"])
    col2.metric("Win Rate %", f"{matrix['win_rate']:.2f}")
    col3.metric("Total P&L (INR)", f"{matrix['total_pnl']:.2f}")
    col4.metric("Winning / Losing", f"{matrix['wins']} / {matrix['losses']}")
    col5.metric("Max Drawdown", f"{matrix['max_drawdown']:.2f}")

    st.markdown("---")
    st.subheader("Trade Log")

    if trades:
        df = pd.DataFrame(trades)
        for col in ["entry_time", "exit_time"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = df.sort_values(by="entry_time", ascending=False)
        st.dataframe(df, use_container_width=True, height=400)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="paper_trades.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades yet. Strategy is monitoring live prices.")


def render_backtest_tab() -> None:
    st.subheader("Backtesting (Fyers history API)")

    symbol_name = st.selectbox(
        "Nifty-50 Symbol",
        sorted(NIFTY_50_SYMBOLS.keys()),
        index=0,
        key="bt_symbol",
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "3m", "5m", "15m", "60m", "120m", "180m", "240m", "1D"],
        index=2,
        key="bt_timeframe",
    )

    today = date.today()
    default_start = today - timedelta(days=30)

    date_range = st.date_input(
        "Date Range (From / To)",
        (default_start, today),
        key="bt_date_range",
    )

    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.warning("Please select both start and end dates.")
        return

    start_date, end_date = date_range

    if st.button("Run Backtest", key="bt_run"):
        with st.spinner("Fetching historical data and running backtest..."):
            trades_df = run_backtest(symbol_name, timeframe, start_date, end_date)

        if trades_df is None or trades_df.empty:
            st.warning("No trades generated for the selected configuration.")
            return

        st.success(f"Backtest completed with {len(trades_df)} trades.")

        matrix = compute_pnl_matrix(trades_df.to_dict("records"))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Trades", matrix["total_trades"])
        col2.metric("Win Rate %", f"{matrix['win_rate']:.2f}")
        col3.metric("Total P&L (INR)", f"{matrix['total_pnl']:.2f}")
        col4.metric("Winning / Losing", f"{matrix['wins']} / {matrix['losses']}")
        col5.metric("Max Drawdown", f"{matrix['max_drawdown']:.2f}")

        if "cum_pnl" in trades_df.columns:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=trades_df["exit_time"],
                        y=trades_df["cum_pnl"],
                        mode="lines",
                        name="Cumulative P&L",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Time",
                yaxis_title="P&L (INR)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Backtest Trades")
        show_df = trades_df.copy()
        for col in ["entry_time", "exit_time"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].astype(str)

        st.dataframe(show_df, use_container_width=True, height=400)

        csv = show_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="backtest_results.csv",
            mime="text/csv",
        )


def render_signals_tab() -> None:
    st.subheader("Live Engine Signals (Production Strategies)")

    manager = get_engine_manager()
    if manager is None:
        st.warning("Engine is not running. Check FYERS credentials / access token.")
    else:
        status: Dict[str, Any] = {}
        try:
            status = manager.get_status() or {}
        except Exception as e:
            st.error(f"Error fetching engine status: {e}")

        stats = status.get("statistics", {}) if isinstance(status, dict) else {}
        cols = st.columns(3)
        with cols[0]:
            st.metric("Engine Running", "Yes" if status.get("is_running") else "No")
        with cols[1]:
            st.metric("Signals Generated", stats.get("signals_generated", 0))
        with cols[2]:
            st.metric("Errors", stats.get("errors", 0))

        engine_signals: List[Dict[str, Any]] = []
        try:
            engine_signals = manager.get_signals(limit=50) or []
        except Exception as e:
            st.error(f"Error fetching engine signals: {e}")

        if engine_signals:
            df_engine = pd.DataFrame(engine_signals)
            for col in ["signal_timestamp", "timestamp_generated"]:
                if col in df_engine.columns:
                    df_engine[col] = df_engine[col].astype(str)

            st.dataframe(df_engine, use_container_width=True, height=350)

            csv_engine = df_engine.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Engine Signals CSV",
                data=csv_engine,
                file_name="engine_signals.csv",
                mime="text/csv",
            )
        else:
            st.info("Engine is running but no signals have been generated yet.")


def render_analytics_and_backtesting_tab() -> None:
    """Aggregate analytics, paper trade log, backtesting, and engine signals."""

    sub_trades, sub_backtest, sub_signals = st.tabs(
        ["Paper Trade Analytics", "Backtesting", "Engine Signals"]
    )

    with sub_trades:
        render_trades_tab()

    with sub_backtest:
        render_backtest_tab()

    with sub_signals:
        render_signals_tab()


def main() -> None:
    _apply_dark_theme()
    _init_session_state()

    st.title("Nifty-50 Algorithmic Trading Terminal")

    get_ws_manager()
    # Start LiveEngine (production strategies) in background
    get_engine_manager()

    # Header KPIs row
    trades = get_trades_with_pnl_snapshot()
    matrix = compute_pnl_matrix(trades)

    k1, k2, k3, k4, k5 = st.columns(5)
    pnl_val = matrix["total_pnl"]
    pnl_color_class = "kpi-green" if pnl_val >= 0 else "kpi-red"

    with k1:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Live P&amp;L (INR)</div>
                <div class="kpi-value {pnl_color_class}">{pnl_val:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Win Rate %</div>
                <div class="kpi-value kpi-neutral">{matrix['win_rate']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Total Trades</div>
                <div class="kpi-value kpi-neutral">{matrix['total_trades']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Winning / Losing Trades</div>
                <div class="kpi-value kpi-neutral">{matrix['wins']} / {matrix['losses']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Max Drawdown</div>
                <div class="kpi-value kpi-neutral">{matrix['max_drawdown']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    status_cols = st.columns(3)
    with status_cols[0]:
        st.caption(f"Tracked symbols: {len(NIFTY_50_SYMBOLS)}")
    with status_cols[1]:
        st.caption(f"Price buffer size: {len(get_price_snapshot())}")
    with status_cols[2]:
        st.caption(f"Open positions: {len(get_open_positions_snapshot())}")

    tab_live, tab_analytics = st.tabs(["Live Dashboard", "Analytics & Backtesting"])

    with tab_live:
        render_live_tab()

    with tab_analytics:
        render_analytics_and_backtesting_tab()


if __name__ == "__main__":
    main()

