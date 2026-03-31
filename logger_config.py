"""
Logging Configuration Module
Sets up comprehensive logging for the live trading system with PythonAnywhere support.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any
import sys

# Import cloud_utils for path management
try:
    from cloud_utils import get_logs_path, is_pythonanywhere, cloud_logger
    _cloud_utils_available = True
except ImportError:
    _cloud_utils_available = False

class LoggerConfig:
    """Configure logging for the live trading system"""
    
    @staticmethod
    def setup_logging(config: Dict = None):
        """Setup logging configuration"""
        if config is None:
            config = LoggerConfig._get_default_config()
            
        # Use cloud_utils paths if available
        if _cloud_utils_available:
            logs_dir = get_logs_path()
            log_file = os.path.join(logs_dir, os.path.basename(config.get('file', 'live_engine.log')))
            is_pa = is_pythonanywhere()
        else:
            log_file = config.get('file', 'logs/live_engine.log')
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            is_pa = False
        
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter with IST timezone for PythonAnywhere
        if is_pa:
            formatter = logging.Formatter(
                '%(asctime)s IST - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
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
        
        # Console handler (disable on PythonAnywhere production to reduce overhead)
        if config.get('console', True) and not is_pa:
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
            'live_engine',
            'web_main',
            'cloud_deploy'
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
            
        # Log startup message
        startup_logger = logging.getLogger('cloud_deploy')
        if is_pa:
            startup_logger.info("Mini Simon logging configured for PythonAnywhere")
        else:
            startup_logger.info("Mini Simon logging configured for local development")
            
    @staticmethod
    def _get_default_config() -> Dict:
        """Get default logging configuration"""
        # Use cloud paths if available
        if _cloud_utils_available:
            logs_dir = get_logs_path()
            log_file = os.path.join(logs_dir, 'live_engine.log')
            is_pa = is_pythonanywhere()
        else:
            log_file = 'logs/live_engine.log'
            is_pa = False
            
        return {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': log_file,
            'max_size_mb': 100,
            'backup_count': 5,
            'console': not is_pa  # Disable console on PythonAnywhere
        }

# Initialize logging when module is imported
LoggerConfig.setup_logging()
