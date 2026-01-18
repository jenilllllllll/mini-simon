"""
Logging Configuration Module
Sets up comprehensive logging for the live trading system
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any
import sys

class LoggerConfig:
    """Configure logging for the live trading system"""
    
    @staticmethod
    def setup_logging(config: Dict = None):
        """Setup logging configuration"""
        if config is None:
            config = LoggerConfig._get_default_config()
            
        # Create logs directory
        log_file = config.get('file', 'live_engine.log')
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.get('max_size_mb', 100) * 1024 * 1024,
            backupCount=config.get('backup_count', 5)
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        if config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            
        # Set specific logger levels
        loggers_to_configure = [
            'live_data_feed',
            'feature_engine',
            'live_strategy_runner',
            'live_signal_aggregator',
            'signal_store',
            'live_engine'
        ]
        
        for logger_name in loggers_to_configure:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, config.get('level', 'INFO')))
            
        # Suppress noisy third-party loggers
        noisy_loggers = [
            'urllib3.connectionpool',
            'websocket',
            'fyers_apiv3',
            'pandas',
            'numpy'
        ]
        
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            
    @staticmethod
    def _get_default_config() -> Dict:
        """Get default logging configuration"""
        return {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/live_engine.log',
            'max_size_mb': 100,
            'backup_count': 5,
            'console': True
        }

# Initialize logging when module is imported
LoggerConfig.setup_logging()
