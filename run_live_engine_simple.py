"""
Simple Live Engine Runner
Runs the live engine with REST API polling for real-time data
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from logger_config import setup_logging
from live_engine import LiveEngine

def main():
    """Main function to run the live engine"""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Mini-Simon Live Engine (Simple Mode)...")
        
        # Load configuration
        config = Config()
        
        # Validate configuration
        if not config.validate_config():
            logger.error("Configuration validation failed")
            return False
        
        # Create engine
        engine = LiveEngine(config.config_data)
        
        # Initialize engine
        if not engine.initialize():
            logger.error("Engine initialization failed")
            return False
        
        # Start engine
        if not engine.start():
            logger.error("Engine start failed")
            return False
        
        # Keep running until interrupted
        logger.info("Engine is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                # Print status every 30 seconds
                status = engine.get_engine_status()
                logger.info(f"Status: Running={status['is_running']}, "
                          f"Signals Generated={status['statistics']['signals_generated']}, "
                          f"Errors={status['statistics']['errors']}")
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        
        # Stop engine
        engine.stop()
        logger.info("Engine stopped successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
