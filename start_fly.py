#!/usr/bin/env python3
"""
Fly.io Start Script for Mini Simon
Starts the FastAPI server with proper configuration for 24/7 operation
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('fly_startup')

def main():
    logger.info("=" * 60)
    logger.info("Starting Mini Simon on Fly.io")
    logger.info("=" * 60)
    
    # Verify environment
    port = int(os.getenv('PORT', '8080'))
    logger.info(f"Port: {port}")
    
    # Check critical environment variables
    required_vars = ['FYERS_APP_ID', 'FYERS_ACCESS_TOKEN']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Set them using: fly secrets set VAR_NAME=value")
        sys.exit(1)
    
    logger.info("Environment variables: OK")
    logger.info(f"FYERS_APP_ID: {os.getenv('FYERS_APP_ID')[:10]}...")
    logger.info(f"FYERS_ACCESS_TOKEN: {os.getenv('FYERS_ACCESS_TOKEN')[:20]}...")
    
    # Create necessary directories
    from cloud_utils import get_logs_path, get_signals_path, get_trades_path, get_data_path
    paths = {
        'logs': get_logs_path(),
        'signals': get_signals_path(),
        'trades': get_trades_path(),
        'data': get_data_path()
    }
    
    for name, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory {name}: {path}")
    
    # Import and start the FastAPI app
    logger.info("Starting FastAPI server...")
    
    import uvicorn
    from web_main import app
    
    logger.info("FastAPI app loaded successfully")
    logger.info("=" * 60)
    logger.info(f"Server will start on 0.0.0.0:{port}")
    logger.info("=" * 60)
    
    # Start uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        reload=False  # Disable reload for production
    )

if __name__ == "__main__":
    main()
