"""
Run Live Engine - Simple launcher script
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_engine import EngineManager
from config import get_config
from logger_config import LoggerConfig
import logging

def main():
    """Main launcher function"""
    # Setup logging
    LoggerConfig.setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Mini-Simon Live Engine...")
    
    try:
        # Load configuration
        config = get_config()
        
        # Validate configuration
        if not config.validate_config():
            logger.error("❌ Configuration validation failed")
            return False
            
        # Apply environment overrides
        config.apply_env_overrides()
        
        # Create engine manager
        manager = EngineManager()
        
        # Start engine
        if manager.start_engine():
            logger.info("✅ Engine started successfully!")
            logger.info("📊 Monitoring live signals...")
            logger.info("🛑 Press Ctrl+C to stop")
            
            try:
                # Keep running until interrupted
                while True:
                    import time
                    time.sleep(10)
                    
                    # Print periodic status
                    status = manager.get_status()
                    if status.get('statistics'):
                        stats = status['statistics']
                        logger.info(f"📈 Signals Generated: {stats.get('signals_generated', 0)}, "
                                  f"Signals Stored: {stats.get('signals_stored', 0)}, "
                                  f"Errors: {stats.get('errors', 0)}")
                        
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested by user")
                
        else:
            logger.error("❌ Failed to start engine")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False
        
    finally:
        # Cleanup
        if 'manager' in locals():
            manager.stop_engine()
            logger.info("🧹 Engine stopped and cleaned up")
            
    logger.info("👋 Mini-Simon Live Engine shutdown complete")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
