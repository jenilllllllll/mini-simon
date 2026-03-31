"""
Configuration Management Module
Handles all configuration settings for the live trading system and dashboard.

NOTE: Sensitive values like Fyers app_id and access_token are intentionally
not hard-coded here. They are read from environment variables so that this
file can be safely committed to GitHub.

Set these environment variables (or in a .env file for local development):

- FYERS_APP_ID
- FYERS_ACCESS_TOKEN

For PythonAnywhere deployment:
1. Create a .env file in the project root
2. Set environment variables in the PythonAnywhere dashboard
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Import cloud utilities for path management
try:
    from cloud_utils import (
        get_logs_path, get_signals_path, get_trades_path, 
        get_data_path, get_fyers_credentials, get_websocket_config,
        get_trading_config, is_pythonanywhere, cloud_logger
    )
    _cloud_utils_available = True
except ImportError:
    _cloud_utils_available = False

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with support for multiple formats."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file or use defaults."""

        if self.config_file and Path(self.config_file).exists():
            try:
                self.config_data = self._load_from_file(self.config_file)
                logger.info("Loaded configuration from %s", self.config_file)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Error loading config file: %s", exc)
                self.config_data = self._get_default_config()
        else:
            self.config_data = self._get_default_config()
            logger.info("Using default configuration")

    def _load_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file."""

        path = Path(filepath)
        with path.open("r", encoding="utf-8") as fp:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(fp) or {}
            if path.suffix.lower() == ".json":
                return json.load(fp)
            msg = f"Unsupported config format: {path.suffix}"
            raise ValueError(msg)

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration used when no external file is provided.

        Fyers credentials are read from environment variables so that secrets
        are not stored in the repository.
        """

        app_id = os.getenv("FYERS_APP_ID", "YOUR_APP_ID")
        access_token = os.getenv("FYERS_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

        # Use cloud_utils paths if available for PythonAnywhere compatibility
        if _cloud_utils_available:
            log_path = get_logs_path()
            log_file = os.path.join(log_path, "live_engine.log")
            ws_config = get_websocket_config()
            trading_config = get_trading_config()
        else:
            log_path = "logs"
            log_file = "logs/live_engine.log"
            ws_config = {
                'max_reconnect_attempts': int(os.getenv('WS_MAX_RECONNECT_ATTEMPTS', '10')),
                'reconnect_delay': int(os.getenv('WS_RECONNECT_DELAY', '5')),
                'heartbeat_interval': int(os.getenv('WS_HEARTBEAT_INTERVAL', '30')),
                'tick_timeout': int(os.getenv('WS_TICK_TIMEOUT', '90')),
            }
            trading_config = {'enable_mcx': True}

        return {
            # Data Feed Configuration
            "data_feed": {
                "app_id": app_id,
                "access_token": access_token,
                "symbols": [
                    "RELIANCE",
                    "TCS", 
                    "INFY",
                    "HDFCBANK",
                    "ICICIBANK",
                    "KOTAKBANK",
                    "LT",
                    "ITC",
                    "SBIN",
                    "HINDUNILVR",
                    "AXISBANK",
                    "BAJFINANCE",
                    "ASIANPAINT",
                    "MARUTI",
                    "SUNPHARMA",
                    "WIPRO",
                    "ULTRACEMCO",
                    "NESTLEIND",
                    "POWERGRID",
                    "BAJAJFINSV",
                    "TECHM",
                    "NTPC",
                    "GRASIM",
                    "JSWSTEEL",
                    "HCLTECH",
                    "TATAMOTORS",
                    "DRREDDY",
                    "CIPLA",
                    "ONGC",
                    "HDFCLIFE",
                    "DIVISLAB",
                    "HEROMOTOCO",
                    "BRITANNIA",
                    "BPCL",
                    "COALINDIA",
                    "ADANIENT",
                    "ADANIPORTS",
                    "INDUSINDBK",
                    "BAJAJ-AUTO",
                    "EICHERMOT",
                    "TATACONSUM",
                    "HINDALCO",
                    "APOLLOHOSP",
                    "TATASTEEL",
                    "M&M",
                    "BHARTIARTL",
                    "SHRIRAMFIN",
                    "JIOFINANCE",
                ],
                "timeframes": ["5m", "15m", "60m"],
                "log_path": log_path,
                "reconnect_attempts": ws_config['max_reconnect_attempts'],
                "reconnect_delay": ws_config['reconnect_delay'],
                "heartbeat_interval": ws_config['heartbeat_interval'],
                "tick_timeout": ws_config['tick_timeout'],
                "enable_mcx": trading_config.get('enable_mcx', True),
            },
            # Additional sections kept minimal for dashboard usage; extend as
            # needed for the full engine.
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": log_file,
                "max_size_mb": 100,
                "backup_count": 5,
                "console": not _cloud_utils_available or not is_pythonanywhere(),
            },
            # Strategy Configuration
            "strategy_runner": {
                "strategy_weights": {
                    "vol_spike": 0.35,
                    "body_imbalance": 0.25,
                    "order_block": 0.25,
                    "stock_burner": 0.15,
                    "ema_scalping_5": 0.40
                },
                "timeframe_weights": {
                    "3m": 0.5, "5m": 0.6, "15m": 0.7,
                    "60m": 1.0, "120m": 1.2, "180m": 1.3,
                    "240m": 1.4, "1D": 1.6
                }
            },
            "vol_spike": {
                "enabled": True,
                "volume_multiplier": 1.5,
                "lookback": 20
            },
            "body_imbalance": {
                "enabled": True,
                "min_body_ratio": 0.5
            },
            "order_block": {
                "enabled": True,
                "atr_period": 14
            },
            "stock_burner": {
                "enabled": True,
                "fast_ema": 9,
                "slow_ema": 21
            },
            "ema_scalping_5": {
                "enabled": True,
                "confidence": 0.70
            },
            "signal_aggregator": {
                "min_confidence_threshold": 0.2,  # Reduced from 0.3 for more signals
                "confluence_threshold": 1        # Reduced from 2 for easier signal generation
            }
        }

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get configuration value using dotted key notation.

        Example: cfg.get("data_feed.app_id")
        """

        parts = key.split(".")
        value: Any = self.config_data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dotted key notation."""

        parts = key.split(".")
        cfg = self.config_data
        for part in parts[:-1]:
            if part not in cfg or not isinstance(cfg[part], dict):
                cfg[part] = {}
            cfg = cfg[part]
        cfg[parts[-1]] = value


# Global configuration instance reused across modules
_config = Config()


def get_config() -> Config:
    """Return global configuration instance."""

    return _config
