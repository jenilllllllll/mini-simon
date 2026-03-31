#!/usr/bin/env python3
"""Background engine runner for PythonAnywhere scheduled tasks.

This script is designed to run as a scheduled task (cron job) on PythonAnywhere
to keep the Mini Simon engine running during market hours.

Usage:
    # Run via PythonAnywhere scheduled task:
    # Command: /home/YOUR_USERNAME/.virtualenvs/mini-simon/bin/python /home/YOUR_USERNAME/mini-simon/background_engine.py
    # Schedule: */5 9-15 * * 1-5 (every 5 minutes, 9 AM-3 PM, Mon-Fri)
"""

import os
import sys
import time
import logging
from datetime import datetime, time as dt_time
from typing import Optional

# ============================================================================
# Setup Paths (Required for PythonAnywhere)
# ============================================================================
# Detect if running on PythonAnywhere and set up paths
if '/home/' in os.getcwd() or 'pythonanywhere' in os.getenv('HOSTNAME', ''):
    # PythonAnywhere environment
    username = os.getenv('USER', os.getenv('PYTHONANYWHERE_USERNAME', 'your_username'))
    project_path = f'/home/{username}/mini-simon'
else:
    # Local development
    project_path = os.path.dirname(os.path.abspath(__file__))

if project_path not in sys.path:
    sys.path.insert(0, project_path)

# ============================================================================
# Load Environment Variables
# ============================================================================
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_path, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# ============================================================================
# Setup Logging
# ============================================================================
try:
    from cloud_utils import get_logs_path, setup_cloud_logging, is_pythonanywhere
    cloud_logger = setup_cloud_logging()
    _cloud_utils_available = True
except ImportError:
    _cloud_utils_available = False
    # Basic logging fallback
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    cloud_logger = logging.getLogger('background_engine')

logger = logging.getLogger('background_engine')

# ============================================================================
# Timezone Setup (Strict IST for Indian Markets)
# ============================================================================
try:
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
    _pytz_available = True
except ImportError:
    _pytz_available = False
    # Fallback - use system local time (may not be accurate)
    IST = None
    logger.warning("pytz not available, using system local time")

def now_ist() -> datetime:
    """Get current time in IST."""
    if _pytz_available:
        return datetime.now(IST)
    return datetime.now()

# ============================================================================
# Market Hours Check
# ============================================================================
def is_equity_market_open() -> bool:
    """Check if NSE equity market is open.
    
    Market Hours: 9:15 AM - 3:30 PM IST, Monday to Friday
    """
    now = now_ist()
    weekday = now.weekday()
    current_time = now.time()
    
    # Weekdays only (Monday=0 to Friday=4)
    if weekday > 4:
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    
    return market_open <= current_time <= market_close

def is_mcx_market_open() -> bool:
    """Check if MCX commodity market is open.
    
    Market Hours: 9:00 AM - 11:30 PM IST, Monday to Friday
    """
    now = now_ist()
    weekday = now.weekday()
    current_time = now.time()
    
    # Weekdays only
    if weekday > 4:
        return False
    
    # MCX hours: 9:00 AM to 11:30 PM IST
    mcx_open = dt_time(9, 0)
    mcx_close = dt_time(23, 30)
    
    return mcx_open <= current_time <= mcx_close

def is_any_market_open() -> bool:
    """Check if any market (equity or MCX) is open."""
    return is_equity_market_open() or is_mcx_market_open()

# ============================================================================
# Engine Components
# ============================================================================
_engine_running = False
_data_feed = None

def start_engine() -> bool:
    """Initialize and start the live data feed engine.
    
    Returns:
        True if engine started successfully, False otherwise.
    """
    global _engine_running, _data_feed
    
    try:
        logger.info("Initializing Mini Simon engine...")
        
        # Import required modules
        from config import get_config
        from live_data_feed import LiveDataFeed
        
        # Get configuration
        cfg = get_config()
        data_feed_config = cfg.config_data.get('data_feed', {})
        
        # Validate credentials
        app_id = data_feed_config.get('app_id', '')
        access_token = data_feed_config.get('access_token', '')
        
        if not app_id or app_id == 'YOUR_APP_ID':
            logger.error("Fyers App ID not configured")
            return False
            
        if not access_token or access_token == 'YOUR_ACCESS_TOKEN':
            logger.error("Fyers Access Token not configured")
            return False
        
        logger.info(f"Starting engine with {len(data_feed_config.get('symbols', []))} symbols")
        
        # Initialize data feed
        _data_feed = LiveDataFeed(data_feed_config)
        
        # Start the feed
        _data_feed.start()
        _engine_running = True
        
        logger.info("Mini Simon engine started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start engine: {e}")
        _engine_running = False
        return False

def stop_engine():
    """Stop the engine cleanly."""
    global _engine_running, _data_feed
    
    try:
        if _data_feed:
            logger.info("Stopping Mini Simon engine...")
            _data_feed.stop()
            _data_feed = None
        
        _engine_running = False
        logger.info("Engine stopped")
        
    except Exception as e:
        logger.error(f"Error stopping engine: {e}")

def check_engine_health() -> dict:
    """Check engine health status.
    
    Returns:
        Dictionary with health metrics.
    """
    global _engine_running, _data_feed
    
    status = {
        'running': _engine_running,
        'ws_connected': False,
        'market_open': is_any_market_open(),
        'timestamp': now_ist().isoformat()
    }
    
    if _data_feed and _engine_running:
        try:
            status['ws_connected'] = _data_feed.is_connected()
        except Exception as e:
            logger.warning(f"Could not check connection status: {e}")
    
    return status

# ============================================================================
# Main Run Loop
# ============================================================================
def run_once(max_runtime_minutes: int = 5):
    """Run the engine for a single scheduled task execution.
    
    This function is designed to be called by PythonAnywhere scheduled tasks.
    It starts the engine if not running, keeps it alive for the specified
    duration, then cleanly shuts down.
    
    Args:
        max_runtime_minutes: Maximum runtime in minutes before exiting.
    """
    global _engine_running
    
    logger.info("=" * 60)
    logger.info("Mini Simon Background Engine - Starting Run")
    logger.info(f"Current IST Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Market Open: {is_any_market_open()}")
    logger.info("=" * 60)
    
    # Check if market is open
    if not is_any_market_open():
        logger.info("Market is closed. Skipping engine run.")
        return
    
    # Start engine if not running
    if not _engine_running:
        if not start_engine():
            logger.error("Failed to start engine. Exiting.")
            return
    
    # Run for the specified duration
    start_time = time.time()
    max_runtime_seconds = max_runtime_minutes * 60
    
    try:
        while time.time() - start_time < max_runtime_seconds:
            # Check if market is still open
            if not is_any_market_open():
                logger.info("Market closed during run. Stopping engine.")
                break
            
            # Check engine health
            health = check_engine_health()
            
            if not health['ws_connected']:
                logger.warning("WebSocket disconnected. Attempting to maintain connection...")
                # The WebSocket feed has its own reconnection logic
                # We just log the status here
            
            # Log status periodically (every 30 seconds)
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:
                logger.info(f"Engine running... Elapsed: {elapsed}s, WS Connected: {health['ws_connected']}")
            
            # Sleep briefly to prevent CPU spinning
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Stopping...")
    except Exception as e:
        logger.error(f"Error during engine run: {e}")
    finally:
        stop_engine()
        logger.info("Run completed. Engine stopped.")

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    # Default: Run for 5 minutes (suitable for 5-minute scheduled tasks)
    # This allows the scheduled task to restart the engine if it crashes
    run_once(max_runtime_minutes=5)
