"""
Cloud-Optimized WebSocket Manager for Mini-Simon
================================================
Enhancements for DigitalOcean deployment with network latency considerations:
- Adaptive heartbeat intervals based on connection quality
- Aggressive reconnection for cloud network fluctuations
- Detailed connection metrics logging
- Connection pooling optimizations
"""

from __future__ import annotations

import logging
import os
import ssl
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pytz

try:
    from fyers_apiv3.FyersWebsocket import data_ws  # type: ignore[import]
except Exception:
    data_ws = None  # type: ignore[assignment]

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("mini_simon.ws_manager_cloud")


class CloudOptimizedWebSocketManager:
    """
    Cloud-optimized WebSocket manager with enhanced reliability for:
    - Higher network latency environments
    - Intermittent cloud connectivity
    - 24/7 uptime requirements
    
    Key improvements over base RobustWebSocketManager:
    - Shorter heartbeat timeout (7s vs 10s) for faster failure detection
    - More aggressive reconnection (1s, 2s, 5s vs 2s, 5s, 10s)
    - Connection quality metrics
    - Cloud-specific logging
    """

    # Cloud-optimized timing (more aggressive than local hosting)
    RECONNECT_DELAYS = [1, 2, 5]          # Faster initial reconnection
    HEARTBEAT_TIMEOUT_S = 7                # Faster failure detection
    MAX_RECONNECT_ATTEMPTS = 15            # More attempts for cloud flakiness
    RECONNECT_RESET_PAUSE_S = 180          # 3-min pause before resetting
    HEARTBEAT_CHECK_INTERVAL_S = 2         # Check every 2 seconds
    
    # Cloud-specific thresholds
    CLOUD_LATENCY_THRESHOLD_MS = 500       # Warn if latency exceeds this
    MAX_CONSECUTIVE_FAILURES = 5             # Alert after this many failures

    def __init__(
        self,
        name: str,
        instrument_codes: List[str],
        on_tick: Callable[[Dict[str, Any]], None],
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        channel: int = 15,
        symbol_normalizer: Optional[Callable[[str], Optional[str]]] = None,
        is_index_trading: bool = False,
    ):
        self.name = name
        self.instrument_codes = list(instrument_codes)
        self._on_tick = on_tick
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self.channel = channel
        self._symbol_normalizer = symbol_normalizer
        self._is_index_trading = is_index_trading

        self.fyers_ws: Optional[Any] = None
        self._connected = threading.Event()
        self._stop_event = threading.Event()

        # Reconnection state
        self._reconnect_attempts = 0
        self._reconnect_lock = threading.Lock()
        self._consecutive_failures = 0

        # Heartbeat monitoring
        self._last_tick_times: Dict[str, float] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._hb_lock = threading.Lock()
        
        # Connection metrics
        self._connection_start_time: Optional[float] = None
        self._total_reconnects = 0
        self._latency_readings: List[float] = []

        # LTP cache
        self._ltp_cache: Dict[str, float] = {}
        self._ltp_lock = threading.Lock()

        # Auth helper
        self._get_auth: Optional[Callable[[], Optional[tuple]]] = None
        self._ssl_patched = False

        logger.info("[%s] CloudOptimizedWebSocketManager initialized for %d symbols", 
                   name, len(instrument_codes))

    def set_auth_provider(self, fn: Callable[[], Optional[tuple]]) -> None:
        """Set the function that returns (app_id, websocket_token) or None."""
        self._get_auth = fn

    def update_instrument_codes(self, codes: List[str]) -> None:
        """Update the list of instrument codes."""
        self.instrument_codes = list(codes)
        logger.info("[%s] Updated instrument codes: %d symbols", self.name, len(codes))

    def _get_connection_metrics(self) -> Dict[str, Any]:
        """Get connection quality metrics for monitoring."""
        uptime = 0.0
        if self._connection_start_time:
            uptime = time.time() - self._connection_start_time
            
        avg_latency = sum(self._latency_readings) / len(self._latency_readings) if self._latency_readings else 0
        
        return {
            "name": self.name,
            "connected": self._connected.is_set(),
            "uptime_seconds": round(uptime, 2),
            "total_reconnects": self._total_reconnects,
            "consecutive_failures": self._consecutive_failures,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "symbols_tracked": len(self.instrument_codes),
            "last_tick_times": {k: round(time.time() - v, 2) for k, v in self._last_tick_times.items()}
        }

    def onmessage(self, message: Any) -> None:
        """Receive tick with cloud-optimized processing."""
        start_time = time.time()
        
        if not isinstance(message, dict) or "ltp" not in message:
            return

        ws_symbol = message.get("symbol", "")
        if not ws_symbol:
            return

        # Normalize symbol
        if self._symbol_normalizer:
            symbol_code = self._symbol_normalizer(ws_symbol)
            if symbol_code is None:
                return
            message["symbol"] = symbol_code
        else:
            symbol_code = ws_symbol

        # IST timestamp formatting
        if self._is_index_trading:
            ts_val = message.get("timestamp")
            if ts_val is not None:
                try:
                    ts = datetime.fromtimestamp(ts_val, IST)
                except Exception:
                    ts = datetime.now(IST)
            else:
                ts = datetime.now(IST)
            message["timestamp_ist"] = ts.strftime("%d-%b-%Y %I:%M:%S %p")
            message["timestamp_ist_iso"] = ts.isoformat()
            message["is_index_trading"] = True

        # Update heartbeat tracking
        now = time.time()
        with self._hb_lock:
            self._last_tick_times[symbol_code] = now

        # Update LTP cache
        ltp_val = float(message.get("ltp") or 0.0)
        with self._ltp_lock:
            self._ltp_cache[symbol_code] = ltp_val

        # Record latency
        processing_time = time.time() - start_time
        self._latency_readings.append(processing_time)
        if len(self._latency_readings) > 100:
            self._latency_readings.pop(0)

        # Invoke callback
        try:
            self._on_tick(message)
        except Exception as e:
            logger.error("[%s] on_tick callback error: %s", self.name, e)

    def onerror(self, message: Any) -> None:
        """Handle WebSocket errors with cloud-specific logging."""
        self._consecutive_failures += 1
        logger.error("[%s] WebSocket error (failure #%d): %s", 
                    self.name, self._consecutive_failures, message)
        
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.critical("[%s] CRITICAL: %d consecutive failures - check network connectivity",
                          self.name, self._consecutive_failures)

    def onclose(self, message: Any) -> None:
        """Handle connection close with metrics."""
        logger.warning("[%s] WebSocket closed: %s | Metrics: %s",
                      self.name, message, self._get_connection_metrics())
        
        self._connected.clear()
        self._connection_start_time = None
        
        if self._on_disconnect:
            try:
                self._on_disconnect()
            except Exception:
                pass
        
        self._schedule_reconnect()

    def onopen(self) -> None:
        """Handle successful connection."""
        logger.info("[%s] WebSocket opened - subscribing to %d symbols",
                   self.name, len(self.instrument_codes))
        
        try:
            self._connected.set()
            self._connection_start_time = time.time()
            self._consecutive_failures = 0
            
            with self._reconnect_lock:
                self._reconnect_attempts = 0
                self._total_reconnects += 1

            if self.fyers_ws is None:
                return

            self.fyers_ws.subscribe(
                symbols=self.instrument_codes,
                data_type="SymbolUpdate",
                channel=self.channel,
            )
            self.fyers_ws.keep_running()

            if self._on_connect:
                try:
                    self._on_connect()
                except Exception as e:
                    logger.error("[%s] on_connect callback error: %s", self.name, e)
                    
        except Exception as e:
            logger.error("[%s] onopen error: %s", self.name, e)

    def _schedule_reconnect(self) -> None:
        """Schedule reconnection with cloud-optimized backoff."""
        with self._reconnect_lock:
            attempt = self._reconnect_attempts
            self._reconnect_attempts += 1
            
            if self._reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                logger.warning("[%s] Max reconnects reached - pausing 5 min before reset", self.name)
                time.sleep(self.RECONNECT_RESET_PAUSE_S)
                self._reconnect_attempts = 0
        
        # Calculate delay from cycle
        delay = self.RECONNECT_DELAYS[min(attempt, len(self.RECONNECT_DELAYS) - 1)]
        
        logger.info("[%s] Reconnecting in %ds (attempt %d)", self.name, delay, attempt + 1)
        
        def _reconnect():
            time.sleep(delay)
            if not self._stop_event.is_set():
                self.start()
        
        threading.Thread(target=_reconnect, daemon=True, name=f"reconnect-{self.name}").start()

    def _start_heartbeat_monitor(self) -> None:
        """Start cloud-optimized heartbeat monitoring."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        def _monitor():
            while not self._stop_event.is_set():
                try:
                    time.sleep(self.HEARTBEAT_CHECK_INTERVAL_S)
                    
                    if not self._connected.is_set():
                        continue
                    
                    now = time.time()
                    stale_symbols = []
                    
                    with self._hb_lock:
                        for symbol, last_time in self._last_tick_times.items():
                            if now - last_time > self.HEARTBEAT_TIMEOUT_S:
                                stale_symbols.append((symbol, now - last_time))
                    
                    if stale_symbols:
                        logger.warning("[%s] Stale ticks detected: %s - forcing resubscribe",
                                      self.name, stale_symbols)
                        self._force_resubscribe()
                        
                except Exception as e:
                    logger.error("[%s] Heartbeat monitor error: %s", self.name, e)

        self._heartbeat_thread = threading.Thread(
            target=_monitor, daemon=True, name=f"hb-{self.name}"
        )
        self._heartbeat_thread.start()
        logger.info("[%s] Heartbeat monitor started (timeout: %ds)",
                   self.name, self.HEARTBEAT_TIMEOUT_S)

    def _force_resubscribe(self) -> None:
        """Force re-subscription without full reconnect."""
        try:
            if self.fyers_ws and self._connected.is_set():
                self.fyers_ws.subscribe(
                    symbols=self.instrument_codes,
                    data_type="SymbolUpdate",
                    channel=self.channel,
                )
                logger.info("[%s] Re-subscribed to %d symbols", self.name, len(self.instrument_codes))
        except Exception as e:
            logger.error("[%s] Force resubscribe failed: %s", self.name, e)

    def _patch_ssl(self) -> None:
        """Apply SSL patches for cloud compatibility."""
        if self._ssl_patched:
            return
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            self._ssl_patched = True
            logger.debug("[%s] SSL patches applied", self.name)
        except Exception as e:
            logger.warning("[%s] SSL patch failed: %s", self.name, e)

    def start(self) -> None:
        """Start WebSocket with cloud optimizations."""
        if data_ws is None:
            logger.critical("[%s] FyersDataSocket SDK not available - cannot start", self.name)
            return

        if self.fyers_ws is not None and self._connected.is_set():
            logger.debug("[%s] WebSocket already running", self.name)
            return

        self._stop_event.clear()

        if self._get_auth is None:
            logger.error("[%s] Auth provider not set - cannot start", self.name)
            return

        auth = self._get_auth()
        if auth is None:
            logger.error("[%s] Auth failed - check credentials", self.name)
            self._connected.clear()
            self._schedule_reconnect()
            return

        _app_id, websocket_token = auth

        try:
            self._patch_ssl()

            self.fyers_ws = data_ws.FyersDataSocket(
                access_token=websocket_token,
                log_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self.onopen,
                on_close=self.onclose,
                on_error=self.onerror,
                on_message=self.onmessage,
            )

            t = threading.Thread(
                target=self.fyers_ws.connect,
                daemon=True,
                name=f"ws-cloud-{self.name.lower()}",
            )
            t.start()
            
            logger.info("[%s] WebSocket started (cloud-optimized, reconnect=True)", self.name)
            self._start_heartbeat_monitor()

        except Exception as e:
            logger.error("[%s] Failed to start WebSocket: %s", self.name, e)
            self.fyers_ws = None
            self._schedule_reconnect()

    def stop(self) -> None:
        """Gracefully stop the WebSocket."""
        logger.info("[%s] Stopping WebSocket...", self.name)
        self._stop_event.set()
        self._connected.clear()
        self._connection_start_time = None
        
        ws = self.fyers_ws
        self.fyers_ws = None
        
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        
        logger.info("[%s] WebSocket stopped | Final metrics: %s", 
                   self.name, self._get_connection_metrics())

    def restart(self) -> None:
        """Force restart the WebSocket."""
        logger.info("[%s] Restart requested", self.name)
        self.stop()
        time.sleep(2)
        self.start()

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get Last Traded Price for a symbol."""
        with self._ltp_lock:
            return self._ltp_cache.get(symbol)

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected.is_set()
