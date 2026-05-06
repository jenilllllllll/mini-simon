#!/usr/bin/env python3
"""
Production Token Manager for Mini-Simon
=======================================
Automated Fyers token refresh for 24/7 cloud deployment.

Features:
- Automatic token refresh before expiry (23-hour cycle)
- Headless Selenium automation (no browser window)
- Updates .env file and reloads application
- Discord/Slack notifications on success/failure
- Retry logic with exponential backoff
- Health checks for token validity

Usage:
    python token_manager.py --refresh-now    # Force immediate refresh
    python token_manager.py --daemon         # Run as background daemon
    python token_manager.py --check          # Check token validity

Cron Setup (recommended):
    0 6 * * * /opt/mini-simon/venv/bin/python /opt/mini-simon/deploy/token_manager.py --refresh-now
    (Runs daily at 6 AM IST before market opens at 9:15 AM)
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qs

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/mini-simon/token_manager.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("token_manager")

IST = pytz.timezone("Asia/Kolkata")
APP_DIR = Path("/opt/mini-simon")
ENV_FILE = APP_DIR / ".env"
ACCESS_FILE = APP_DIR / "access.txt"
CREDENTIALS_FILE = APP_DIR / "credentials.py"


def load_credentials() -> Optional[Dict[str, str]]:
    """Load credentials from credentials.py"""
    try:
        sys.path.insert(0, str(APP_DIR))
        import credentials as cd
        return {
            "client_id": cd.client_id,
            "secret_key": cd.secret_key,
            "redirect_uri": cd.redirect_uri,
            "user_name": cd.user_name,
            "pin1": cd.pin1,
            "pin2": cd.pin2,
            "pin3": cd.pin3,
            "pin4": cd.pin4,
            "totp_key": cd.totp_key,
        }
    except Exception as e:
        logger.error(f"Failed to load credentials: {e}")
        return None


def get_current_token() -> Optional[str]:
    """Get current token from .env or access.txt"""
    # Try .env first
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if line.startswith('FYERS_ACCESS_TOKEN='):
                    return line.strip().split('=', 1)[1]
    
    # Try access.txt
    if ACCESS_FILE.exists():
        return ACCESS_FILE.read_text().strip()
    
    return None


def update_env_token(token: str) -> bool:
    """Update .env file with new token"""
    try:
        lines = []
        token_updated = False
        
        if ENV_FILE.exists():
            with open(ENV_FILE, 'r') as f:
                lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.startswith('FYERS_ACCESS_TOKEN='):
                new_lines.append(f'FYERS_ACCESS_TOKEN={token}\n')
                token_updated = True
            else:
                new_lines.append(line)
        
        if not token_updated:
            new_lines.append(f'FYERS_ACCESS_TOKEN={token}\n')
        
        with open(ENV_FILE, 'w') as f:
            f.writelines(new_lines)
        
        # Also update access.txt
        with open(ACCESS_FILE, 'w') as f:
            f.write(token)
        
        # Update PM2 environment
        subprocess.run(['pm2', 'reload', 'ecosystem.config.js', '--update-env'], 
                      cwd=APP_DIR, capture_output=True)
        
        logger.info("Token updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update token: {e}")
        return False


def generate_totp(totp_key: str) -> str:
    """Generate TOTP code"""
    try:
        import pyotp
        totp = pyotp.TOTP(totp_key)
        return totp.now()
    except ImportError:
        logger.error("pyotp not installed. Run: pip install pyotp")
        return ""


def refresh_token_headless(creds: Dict[str, str]) -> Optional[str]:
    """
    Automated token refresh using headless Selenium.
    Runs without opening browser window - suitable for server deployment.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        import pyotp
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Install with: pip install selenium pyotp webdriver-manager")
        return None
    
    from fyers_apiv3 import fyersModel
    
    # Generate auth URL
    session = fyersModel.SessionModel(
        client_id=creds["client_id"],
        secret_key=creds["secret_key"],
        redirect_uri=creds["redirect_uri"],
        response_type="code"
    )
    auth_url = session.generate_authcode()
    
    logger.info("Starting headless browser for token refresh...")
    
    # Setup headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
    
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Navigate to auth URL
        logger.info("Navigating to Fyers login...")
        driver.get(auth_url)
        
        wait = WebDriverWait(driver, 30)
        
        # Step 1: Enter username
        username_field = wait.until(EC.presence_of_element_located((By.ID, "fyers_id")))
        username_field.send_keys(creds["user_name"])
        logger.info("Username entered")
        
        # Step 2: Enter password (PIN)
        pin = f"{creds['pin1']}{creds['pin2']}{creds['pin3']}{creds['pin4']}"
        password_field = driver.find_element(By.ID, "password")
        password_field.send_keys(pin)
        logger.info("PIN entered")
        
        # Step 3: Generate and enter TOTP
        totp = pyotp.TOTP(creds["totp_key"])
        totp_code = totp.now()
        
        totp_field = driver.find_element(By.ID, "totp")
        totp_field.send_keys(totp_code)
        logger.info(f"TOTP entered: {totp_code}")
        
        # Step 4: Click login
        login_btn = driver.find_element(By.ID, "btn_login")
        login_btn.click()
        logger.info("Login submitted")
        
        # Wait for redirect
        time.sleep(5)
        
        # Wait up to 60 seconds for redirect
        for _ in range(12):
            current_url = driver.current_url
            if "auth_code=" in current_url:
                logger.info("Login successful, auth code received")
                break
            time.sleep(5)
        else:
            logger.error("Timeout waiting for redirect")
            # Take screenshot for debugging
            screenshot_path = APP_DIR / "logs" / "login_error.png"
            driver.save_screenshot(str(screenshot_path))
            logger.info(f"Screenshot saved to {screenshot_path}")
            return None
        
        # Extract auth code
        parsed = urlparse(current_url)
        params = parse_qs(parsed.query)
        auth_code = params.get('auth_code', [None])[0]
        
        if not auth_code:
            logger.error("Could not extract auth code from URL")
            return None
        
        logger.info(f"Auth code obtained: {auth_code[:20]}...")
        
        # Exchange for token
        session = fyersModel.SessionModel(
            client_id=creds["client_id"],
            secret_key=creds["secret_key"],
            redirect_uri=creds["redirect_uri"],
            response_type="code",
            grant_type="authorization_code"
        )
        
        session.set_token(auth_code)
        response = session.generate_token()
        access_token = response.get("access_token")
        
        if access_token:
            logger.info(f"Token generated successfully: {access_token[:30]}...")
            return access_token
        else:
            logger.error(f"No token in response: {response}")
            return None
            
    except Exception as e:
        logger.exception(f"Token refresh failed: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def send_notification(message: str, is_error: bool = False) -> None:
    """Send notification to Discord/Slack if configured"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")
    
    if not webhook_url:
        return
    
    try:
        color = 0xFF0000 if is_error else 0x00FF00
        emoji = "🔴" if is_error else "🟢"
        
        payload = {
            "embeds": [{
                "title": f"{emoji} Mini-Simon Token Manager",
                "description": message,
                "color": color,
                "timestamp": datetime.now(IST).isoformat(),
                "footer": {"text": "DigitalOcean Production Server"}
            }]
        }
        
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


def check_token_validity(token: str, client_id: str) -> bool:
    """Check if current token is still valid"""
    try:
        from fyers_apiv3 import fyersModel
        
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=token,
            log_path=str(APP_DIR / "logs")
        )
        
        # Test with profile endpoint
        response = fyers.get_profile()
        
        if response.get("s") == "ok":
            logger.info("Token is valid")
            return True
        else:
            logger.warning(f"Token validation failed: {response}")
            return False
            
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return False


def refresh_token_with_retry(max_retries: int = 3) -> bool:
    """Refresh token with retry logic"""
    creds = load_credentials()
    if not creds:
        send_notification("❌ Failed to load credentials", is_error=True)
        return False
    
    for attempt in range(max_retries):
        logger.info(f"Token refresh attempt {attempt + 1}/{max_retries}")
        
        try:
            new_token = refresh_token_headless(creds)
            
            if new_token:
                # Verify token works
                if check_token_validity(new_token, creds["client_id"]):
                    if update_env_token(new_token):
                        send_notification(
                            f"✅ Token refreshed successfully\n"
                            f"Token: {new_token[:30]}...\n"
                            f"Server: {os.uname().nodename}",
                            is_error=False
                        )
                        return True
                else:
                    logger.error("New token failed validation")
                    
        except Exception as e:
            logger.exception(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            logger.info(f"Waiting {delay}s before retry...")
            time.sleep(delay)
    
    send_notification(
        f"❌ Token refresh failed after {max_retries} attempts\n"
        f"Server: {os.uname().nodename}\n"
        f"Manual intervention required!",
        is_error=True
    )
    return False


def run_daemon() -> None:
    """Run as background daemon - refreshes token every 23 hours"""
    logger.info("Token manager daemon started")
    
    while True:
        now = datetime.now(IST)
        
        # Refresh at 6 AM IST (before market opens at 9:15 AM)
        target_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
        
        if now >= target_time:
            target_time += timedelta(days=1)
        
        wait_seconds = (target_time - now).total_seconds()
        
        logger.info(f"Next refresh scheduled at {target_time} (in {wait_seconds/3600:.1f} hours)")
        
        time.sleep(wait_seconds)
        
        # Perform refresh
        refresh_token_with_retry()


def main():
    parser = argparse.ArgumentParser(description="Mini-Simon Token Manager")
    parser.add_argument("--refresh-now", action="store_true", 
                       help="Force immediate token refresh")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as background daemon")
    parser.add_argument("--check", action="store_true",
                       help="Check current token validity")
    parser.add_argument("--notify-test", action="store_true",
                       help="Send test notification")
    
    args = parser.parse_args()
    
    if args.notify_test:
        send_notification("🧪 Test notification from Token Manager", is_error=False)
        print("Test notification sent")
        return
    
    if args.check:
        token = get_current_token()
        creds = load_credentials()
        
        if not token or not creds:
            print("❌ Missing token or credentials")
            sys.exit(1)
        
        is_valid = check_token_validity(token, creds["client_id"])
        print(f"Token validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
        sys.exit(0 if is_valid else 1)
    
    if args.refresh_now:
        success = refresh_token_with_retry()
        sys.exit(0 if success else 1)
    
    if args.daemon:
        run_daemon()
    
    # Default: check and refresh if needed
    parser.print_help()


if __name__ == "__main__":
    main()
