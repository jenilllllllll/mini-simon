"""
Cloud Configuration Module for PythonAnywhere Deployment
Handles path management, environment detection, and cloud-specific settings.
"""

import os
import logging
from pathlib import Path
from typing import Optional

def is_pythonanywhere() -> bool:
    """Detect if running on PythonAnywhere environment."""
    return bool(
        os.getenv('PYTHONANYWHERE_DOMAIN') or 
        os.getenv('PYTHONANYWHERE_SITE') or
        '/home/' in os.getcwd() and 'pythonanywhere' in os.getenv('HOSTNAME', '')
    )

def get_base_path() -> str:
    """Get the base path for the project.
    
    Returns:
        Absolute path to project root, compatible with local and PythonAnywhere environments.
    """
    if is_pythonanywhere():
        # On PythonAnywhere, use the environment variable or construct from username
        username = os.getenv('PYTHONANYWHERE_USERNAME', os.getenv('USER', 'your_username'))
        return os.getenv('PYTHONANYWHERE_PROJECT_PATH', f'/home/{username}/mini-simon')
    else:
        # Local development - use the directory containing this file
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_logs_path() -> str:
    """Get the logs directory path."""
    base = get_base_path()
    logs_dir = os.path.join(base, 'logs')
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    return logs_dir

def get_signals_path() -> str:
    """Get the signals storage directory path."""
    base = get_base_path()
    signals_dir = os.path.join(base, 'signals')
    Path(signals_dir).mkdir(parents=True, exist_ok=True)
    return signals_dir

def get_trades_path() -> str:
    """Get the trades storage directory path."""
    base = get_base_path()
    trades_dir = os.path.join(base, 'trades')
    Path(trades_dir).mkdir(parents=True, exist_ok=True)
    return trades_dir

def get_data_path() -> str:
    """Get the data directory path."""
    base = get_base_path()
    data_dir = os.path.join(base, 'Data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir

def get_cloud_deploy_log_path() -> str:
    """Get the cloud deployment log file path."""
    logs_dir = get_logs_path()
    return os.path.join(logs_dir, 'cloud_deploy.log')

def setup_cloud_logging() -> logging.Logger:
    """Setup logging specifically for cloud deployment."""
    logger = logging.getLogger('cloud_deploy')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Cloud deploy log file
    log_file = get_cloud_deploy_log_path()
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also add console handler for local debugging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize cloud logger
cloud_logger = setup_cloud_logging()

def get_fyers_credentials() -> dict:
    """Get Fyers API credentials from environment variables.
    
    Returns:
        Dictionary containing app_id and access_token.
        Raises ValueError if credentials are not set.
    """
    app_id = os.getenv('FYERS_APP_ID')
    access_token = os.getenv('FYERS_ACCESS_TOKEN')
    
    if not app_id or app_id == 'YOUR_APP_ID_HERE':
        cloud_logger.error("FYERS_APP_ID not configured in environment")
        raise ValueError("FYERS_APP_ID must be set in .env file")
    
    if not access_token or access_token == 'YOUR_ACCESS_TOKEN_HERE':
        cloud_logger.error("FYERS_ACCESS_TOKEN not configured in environment")
        raise ValueError("FYERS_ACCESS_TOKEN must be set in .env file")
    
    # Handle token format (may include client_id: prefix)
    if ':' in access_token:
        access_token = access_token.split(':', 1)[1]
    
    return {
        'app_id': app_id,
        'access_token': access_token,
        'client_id': os.getenv('FYERS_CLIENT_ID', app_id),
    }

def get_discord_webhook() -> Optional[str]:
    """Get Discord webhook URL from environment."""
    webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if webhook and webhook.startswith('https://discord.com/api/webhooks/'):
        return webhook
    return None

def get_websocket_config() -> dict:
    """Get WebSocket configuration from environment."""
    return {
        'max_reconnect_attempts': int(os.getenv('WS_MAX_RECONNECT_ATTEMPTS', '10')),
        'reconnect_delay': int(os.getenv('WS_RECONNECT_DELAY', '5')),
        'heartbeat_interval': int(os.getenv('WS_HEARTBEAT_INTERVAL', '30')),
        'tick_timeout': int(os.getenv('WS_TICK_TIMEOUT', '90')),
    }

def get_trading_config() -> dict:
    """Get trading configuration from environment."""
    return {
        'paper_trading': os.getenv('PAPER_TRADING_MODE', 'true').lower() == 'true',
        'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '1.0')),
        'account_size': float(os.getenv('ACCOUNT_SIZE', '100000')),
        'enable_mcx': os.getenv('ENABLE_MCX', 'true').lower() == 'true',
    }
