"""
Production entry point for Mini-Simon FastAPI dashboard.
Optimized for cloud deployment on DigitalOcean with PM2 process management.

Run via PM2:
    pm2 start ecosystem.config.js

Or directly (for testing):
    python run_production.py
"""

import os
import sys
import logging
from pathlib import Path

# Force UTF-8 encoding for all I/O operations
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure logs directory exists
logs_dir = Path("/var/log/mini-simon") if os.path.exists("/var/log") else Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Configure basic logging before uvicorn takes over
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "mini-simon.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("mini-simon.production")

try:
    import uvicorn
    from web_main import app
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("Mini-Simon Production Server Starting")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info(f"Logs Directory: {logs_dir}")
    logger.info("=" * 60)
    
    def main() -> None:
        """Run the production UVicorn server."""
        # Get configuration from environment or use defaults
        host = os.getenv("MINI_SIMON_HOST", "127.0.0.1")
        port = int(os.getenv("MINI_SIMON_PORT", "8000"))
        workers = int(os.getenv("MINI_SIMON_WORKERS", "1"))  # PM2 handles multi-process
        
        logger.info(f"Starting server on {host}:{port} with {workers} worker(s)")
        
        uvicorn.run(
            "web_main:app",
            host=host,
            port=port,
            workers=workers,
            reload=False,  # Never use reload in production
            log_level="info",
            access_log=True,
            proxy_headers=True,  # Trust X-Forwarded-* headers from Nginx
            forwarded_allow_ips="*",  # Allow forwarded headers from Nginx
            # Performance tuning for cloud environment
            loop="uvloop",  # Faster event loop on Linux
            http="httptools",  # Faster HTTP parser
            lifespan="on",
            # Graceful shutdown
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30,
        )

    if __name__ == "__main__":
        main()
        
except Exception as e:
    logger.critical(f"Failed to start server: {e}", exc_info=True)
    sys.exit(1)
