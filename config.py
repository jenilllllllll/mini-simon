"""
Configuration Management Module
Handles all configuration settings for the live trading system and dashboard.

NOTE: Sensitive values like Fyers app_id and access_token are intentionally
not hard-coded here. They are read from environment variables so that this
file can be safely committed to GitHub.

Set these environment variables (or Streamlit secrets on Streamlit Cloud):

- FYERS_APP_ID
- FYERS_ACCESS_TOKEN
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with support for multiple formats."""

    def __init__(self, config_file: str | None = None):
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
                ],
                "timeframes": ["5m", "15m", "60m"],
                "log_path": "logs",
                "reconnect_attempts": 5,
                "reconnect_delay": 5,
            },
            # Additional sections kept minimal for dashboard usage; extend as
            # needed for the full engine.
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "live_engine.log",
                "max_size_mb": 100,
                "backup_count": 5,
                "console": True,
            },
        }

    def get(self, key: str, default: Any | None = None) -> Any:
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
