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

active_signals_lock = threading.Lock()
active_signals: List[Dict[str, Any]] = []

fyers_rest_client_lock = threading.Lock()
_fyers_rest_client: Optional[fyersModel.FyersModel] = None

ws_manager_lock = threading.Lock()
_ws_manager: Optional["WebSocketManager"] = None

paper_engine_lock = threading.Lock()
_paper_engine: Optional["PaperTradingEngine"] = None


st.set_page_config(page_title="Nifty-50 Trading Dashboard", layout="wide")


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


def create_signal_from_trade(trade: Dict[str, Any], now: datetime) -> None:
    """Create a manual execution signal (Entry/SL/Target) from a new trade."""

    entry = float(trade["entry_price"])
    direction = trade["direction"]

    if direction == "BUY":
        stop_loss = entry * (1.0 - 0.5 / 100.0)
        target = entry * (1.0 + 1.0 / 100.0)
    else:
        stop_loss = entry * (1.0 + 0.5 / 100.0)
        target = entry * (1.0 - 1.0 / 100.0)

    signal = {
        "symbol": trade["symbol"],
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "time": now,
        "timeframe": trade.get("timeframe"),
        "strategy": trade.get("strategy"),
        "reason": trade.get("reason"),
        "expires_at": now + timedelta(minutes=10),
    }

    with active_signals_lock:
        active_signals.append(signal)


def get_active_signals_snapshot() -> List[Dict[str, Any]]:
    """Return only non-expired active signals (for manual phone execution)."""

    now = datetime.utcnow()
    with active_signals_lock:
        remaining: List[Dict[str, Any]] = []
        for s in active_signals:
            exp = s.get("expires_at")
            if isinstance(exp, datetime) and exp < now:
                continue
            remaining.append(dict(s))

        active_signals.clear()
        active_signals.extend(remaining)

        return [dict(s) for s in remaining]


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


class PaperTradingEngine:
    """Real-time paper trading engine scanning the Price Buffer every second."""

    def __init__(self, scan_interval: float = 1.0):
        self.scan_interval = scan_interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_price: Dict[str, float] = {}
        self._last_signal_time: Dict[str, datetime] = {}
        self._cooldown = timedelta(minutes=5)
        self._stop_loss_pct = 0.005
        self._target_pct = 0.01
        self._min_move_pct = 0.002

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                self.scan()
            except Exception as e:
                print("PaperTradingEngine error:", e)
            time.sleep(self.scan_interval)

    def scan(self) -> None:
        prices = get_price_snapshot()
        now = datetime.utcnow()

        for symbol, data in prices.items():
            ltp_val = data.get("ltp")
            if ltp_val is None:
                continue
            ltp = float(ltp_val)
            if ltp <= 0:
                continue

            prev = self._last_price.get(symbol)
            self._last_price[symbol] = ltp
            if prev is None:
                continue

            move_pct = (ltp - prev) / prev

            self._manage_open_position(symbol, ltp, now)

            last_sig_time = self._last_signal_time.get(symbol)
            if last_sig_time and now - last_sig_time < self._cooldown:
                continue

            if abs(move_pct) < self._min_move_pct:
                continue

            direction = "BUY" if move_pct > 0 else "SELL"
            reason = f"Price moved {move_pct * 100:.2f}% on last tick"

            with positions_lock:
                if symbol in open_positions:
                    continue

            self._open_trade(symbol, direction, ltp, now, reason)
            self._last_signal_time[symbol] = now

    def _open_trade(self, symbol: str, direction: str, price: float, now: datetime, reason: str) -> None:
        trade = {
            "symbol": symbol,
            "direction": direction,
            "qty": 1,
            "entry_price": price,
            "entry_time": now,
            "exit_price": None,
            "exit_time": None,
            "status": "OPEN",
            "timeframe": "1m",
            "strategy": "Momentum_0.2pct",
            "reason": reason,
            "pnl": 0.0,
        }

        with trade_log_lock:
            trade_log.append(trade)

        with positions_lock:
            open_positions[symbol] = trade

        create_signal_from_trade(trade, now)

    def _manage_open_position(self, symbol: str, ltp: float, now: datetime) -> None:
        with positions_lock:
            pos = open_positions.get(symbol)
        if not pos:
            return

        entry = float(pos["entry_price"])
        direction = pos["direction"]
        move_pct = (ltp - entry) / entry
        if direction == "SELL":
            move_pct = -move_pct

        if move_pct <= -self._stop_loss_pct or move_pct >= self._target_pct:
            self._close_trade(symbol, ltp, now)

    def _close_trade(self, symbol: str, ltp: float, now: datetime) -> None:
        with positions_lock:
            pos = open_positions.get(symbol)
        if not pos:
            return

        direction = pos["direction"]
        qty = int(pos["qty"])
        entry_price = float(pos["entry_price"])

        if direction == "BUY":
            pnl_points = ltp - entry_price
        else:
            pnl_points = entry_price - ltp

        pnl_value = pnl_points * qty

        with trade_log_lock:
            pos["status"] = "CLOSED"
            pos["exit_price"] = ltp
            pos["exit_time"] = now
            pos["pnl"] = pnl_value

        with positions_lock:
            open_positions.pop(symbol, None)


def get_paper_engine() -> PaperTradingEngine:
    global _paper_engine
    with paper_engine_lock:
        if _paper_engine is None:
            _paper_engine = PaperTradingEngine()
            _paper_engine.start()
    return _paper_engine


def get_open_positions_snapshot() -> Dict[str, Dict[str, Any]]:
    with positions_lock:
        return {k: dict(v) for k, v in open_positions.items()}


def get_trades_with_pnl_snapshot() -> List[Dict[str, Any]]:
    prices = get_price_snapshot()
    with trade_log_lock:
        trades_copy = [dict(t) for t in trade_log]

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


def backtest_momentum_strategy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 3:
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)

    position: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = float(row["close"])
        prev_close = float(prev["close"])
        move_pct = (close - prev_close) / prev_close

        if position is None:
            if abs(move_pct) >= 0.002:
                direction = "BUY" if move_pct > 0 else "SELL"
                position = {
                    "direction": direction,
                    "entry_price": close,
                    "entry_time": row["date"],
                }
            continue

        entry = float(position["entry_price"])
        direction = position["direction"]
        favored = (close - entry) / entry if direction == "BUY" else (entry - close) / entry

        if favored >= 0.01 or favored <= -0.005:
            exit_price = close
            exit_time = row["date"]
            if direction == "BUY":
                pnl_points = exit_price - entry
            else:
                pnl_points = entry - exit_price
            pnl_val = pnl_points

            trades.append(
                {
                    "symbol": None,
                    "direction": direction,
                    "qty": 1,
                    "entry_price": entry,
                    "entry_time": position["entry_time"],
                    "exit_price": exit_price,
                    "exit_time": exit_time,
                    "status": "CLOSED",
                    "timeframe": None,
                    "strategy": "Momentum_0.2pct",
                    "reason": "Backtest rule hit",
                    "pnl": pnl_val,
                    "points": pnl_points,
                }
            )
            position = None

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    return trades_df


def run_backtest(symbol_name: str, timeframe: str, start: date, end: date) -> pd.DataFrame:
    symbol_code = NIFTY_50_SYMBOLS[symbol_name]
    ohlcv = fetch_history(symbol_code, timeframe, start, end)
    if ohlcv.empty:
        return pd.DataFrame()

    trades_df = backtest_momentum_strategy(ohlcv)
    if trades_df.empty:
        return trades_df

    trades_df["symbol"] = symbol_code
    trades_df["timeframe"] = timeframe
    return trades_df


def render_live_tab() -> None:
    get_ws_manager()
    get_paper_engine()

    st.subheader("Live Dashboard")

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        symbol_name = st.selectbox(
            "Nifty-50 Symbol",
            sorted(NIFTY_50_SYMBOLS.keys()),
            index=0,
        )

    symbol_code = NIFTY_50_SYMBOLS[symbol_name]

    prices = get_price_snapshot()
    latest = prices.get(symbol_code)
    ltp = float(latest["ltp"]) if latest and latest.get("ltp") is not None else None

    with col_sel2:
        if ltp is not None:
            st.metric("LTP", f"{ltp:.2f}")
        else:
            st.metric("LTP", "–")

    df = get_candles_snapshot(symbol_code, limit=200)
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
                    name=symbol_name,
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


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
    st.subheader("Live Manual Signals")

    signals = get_active_signals_snapshot()
    if not signals:
        st.info("No active signals. Signals are generated from the live paper-trading strategy.")
        return

    df = pd.DataFrame(signals)
    for col in ["time", "expires_at"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    st.dataframe(df, use_container_width=True, height=400)


def main() -> None:
    st.title("Nifty-50 Trading Dashboard")

    get_ws_manager()
    get_paper_engine()

    status_cols = st.columns(3)
    with status_cols[0]:
        st.caption(f"Tracked symbols: {len(NIFTY_50_SYMBOLS)}")
    with status_cols[1]:
        st.caption(f"Price buffer size: {len(get_price_snapshot())}")
    with status_cols[2]:
        st.caption(f"Open positions: {len(get_open_positions_snapshot())}")

    tab_live, tab_trades, tab_backtest, tab_signals = st.tabs(
        ["Live Dashboard", "Paper Trade Log", "Backtesting", "Signals"]
    )

    with tab_live:
        render_live_tab()

    with tab_trades:
        render_trades_tab()

    with tab_backtest:
        render_backtest_tab()

    with tab_signals:
        render_signals_tab()


if __name__ == "__main__":
    main()

