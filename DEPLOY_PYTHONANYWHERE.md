# Mini Simon - PythonAnywhere Deployment Guide

This guide provides step-by-step instructions for deploying "Project Mini Simon" on PythonAnywhere while maintaining 100% functional parity with your local setup.

---

## 1. Pre-Deployment Checklist (Local Machine)

### Files Created/Modified for Deployment

| File | Purpose | Status |
|------|---------|--------|
| `wsgi.py` | WSGI entry point for PythonAnywhere | ✅ Created |
| `.env.example` | Environment variables template | ✅ Created |
| `requirements.txt` | Linux-optimized dependencies | ✅ Updated |
| `cloud_utils.py` | Path management & cloud detection | ✅ Created |
| `config.py` | Updated to use .env & cloud_utils | ✅ Modified |
| `logger_config.py` | Cloud-aware logging with cloud_deploy.log | ✅ Modified |
| `signal_store.py` | Uses cloud_utils for paths | ✅ Modified |
| `live_data_feed.py` | Enhanced WebSocket keep-alive | ✅ Modified |

---

## 2. PythonAnywhere Console Setup (Step-by-Step)

### Step 1: Create Virtual Environment

```bash
# SSH into your PythonAnywhere account
# Go to the Bash console

# Create a virtual environment (Python 3.10 recommended)
mkvirtualenv mini-simon --python=/usr/bin/python3.10

# If you need to recreate it later:
# rmvirtualenv mini-simon
# mkvirtualenv mini-simon --python=/usr/bin/python3.10
```

### Step 2: Clone the Repository

```bash
# Navigate to your home directory
cd ~

# Clone the repository (replace with your repo URL)
git clone https://github.com/yourusername/mini-simon.git

# Or if you already have it:
# cd ~/mini-simon
# git pull origin main
```

### Step 3: Install Dependencies

```bash
# Navigate to project directory
cd ~/mini-simon

# Activate virtual environment (if not already active)
workon mini-simon

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(fastapi|uvicorn|fyers|pandas|numpy)"
```

### Step 4: Create Environment Variables File

```bash
cd ~/mini-simon

# Copy the example file
cp .env.example .env

# Edit the .env file with your actual credentials
nano .env
```

**Fill in these required values in `.env`:**

```
# FYERS API CREDENTIALS (Required)
FYERS_APP_ID=YOUR_ACTUAL_APP_ID
FYERS_ACCESS_TOKEN=YOUR_ACTUAL_ACCESS_TOKEN

# DISCORD WEBHOOK (Optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK

# PYTHONANYWHERE CONFIGURATION
PYTHONANYWHERE_USERNAME=your_username
PYTHONANYWHERE_PROJECT_PATH=/home/your_username/mini-simon
```

### Step 5: Create Required Directories

```bash
cd ~/mini-simon

# Create all required directories
mkdir -p logs signals trades Data

# Set proper permissions
chmod 755 logs signals trades Data

# Create .gitignore entries for sensitive data
echo "*.log" >> .gitignore
echo ".env" >> .gitignore
echo "fyersApi.log" >> .gitignore
echo "fyersDataSocket.log" >> .gitignore
echo "fyersRequests.log" >> .gitignore
```

### Step 6: Test the Configuration

```bash
# Activate virtual environment
workon mini-simon

cd ~/mini-simon

# Test if Python can import the modules
python -c "from config import get_config; cfg = get_config(); print('Config loaded:', cfg.get('data_feed.app_id')[:5] + '...')"

# Test cloud_utils
python -c "from cloud_utils import is_pythonanywhere, get_logs_path; print('Is PA:', is_pythonanywhere()); print('Logs path:', get_logs_path())"

# Test FastAPI import
python -c "from web_main import app; print('FastAPI app loaded successfully')"
```

---

## 3. Web App Configuration (PythonAnywhere Dashboard)

### Step 1: Create Web App

1. Go to the **Web** tab in PythonAnywhere dashboard
2. Click **Add a new web app**
3. Select **Manual configuration** (not Flask, not Django)
4. Select **Python 3.10**
5. Click **Next**

### Step 2: Configure WSGI File

1. In the Web tab, find the **WSGI configuration file** section
2. Click on the link to edit the WSGI file
3. Replace the entire contents with:

```python
import sys
import os

# Add your project directory to the sys.path
path = '/home/YOUR_USERNAME/mini-simon'
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variable for PythonAnywhere
os.environ['PYTHONANYWHERE_DOMAIN'] = 'pythonanywhere.com'
os.environ['PYTHONANYWHERE_SITE'] = 'mini-simon'

# Import the FastAPI app
from web_main import app as application

# Wrap with WSGI middleware for PythonAnywhere compatibility
from starlette.middleware.wsgi import WSGIMiddleware
application = WSGIMiddleware(application)
```

**Replace `YOUR_USERNAME` with your actual PythonAnywhere username.**

### Step 3: Configure Virtual Environment

1. In the Web tab, find the **Virtualenv** section
2. Enter the path to your virtual environment:
   ```
   /home/YOUR_USERNAME/.virtualenvs/mini-simon
   ```

### Step 4: Set Environment Variables in Dashboard

1. Go to the **Web** tab
2. Scroll down to **Environment variables**
3. Add these variables:

| Variable | Value |
|----------|-------|
| `FYERS_APP_ID` | Your actual Fyers App ID |
| `FYERS_ACCESS_TOKEN` | Your actual Fyers Access Token |
| `LOG_LEVEL` | INFO |
| `PYTHONANYWHERE_DOMAIN` | pythonanywhere.com |
| `PYTHONANYWHERE_USERNAME` | your_username |

### Step 5: Reload Web App

1. Click the **Reload** button for your web app
2. Wait for the reload to complete
3. Visit your web app URL: `https://yourusername.pythonanywhere.com`

---

## 4. Background Task Setup (24/7 Engine)

PythonAnywhere web workers have timeouts. To keep the engine running 24/7, use a **scheduled task** (cron job).

### Step 1: Create a Background Task Script

Create file `~/mini-simon/background_engine.py`:

```python
#!/usr/bin/env python3
"""Background engine runner for PythonAnywhere scheduled tasks."""

import os
import sys
import time
import logging
from datetime import datetime
import pytz

# Setup paths
sys.path.insert(0, '/home/YOUR_USERNAME/mini-simon')

# Load environment
from dotenv import load_dotenv
load_dotenv('/home/YOUR_USERNAME/mini-simon/.env')

# Setup logging
from logger_config import LoggerConfig
LoggerConfig.setup_logging()

logger = logging.getLogger('background_engine')

# Import engine components
from config import get_config
from live_data_feed import LiveDataFeed

IST = pytz.timezone('Asia/Kolkata')

def is_market_hours():
    """Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
    now = datetime.now(IST)
    weekday = now.weekday()
    current_time = now.time()
    
    # Weekdays only (Monday=0 to Friday=4)
    if weekday > 4:
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    from datetime import time as dt_time
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    
    return market_open <= current_time <= market_close

def run_engine():
    """Run the engine during market hours."""
    logger.info("Starting background engine...")
    
    try:
        # Get configuration
        cfg = get_config()
        data_feed_config = cfg.config_data.get('data_feed', {})
        
        # Initialize data feed
        feed = LiveDataFeed(data_feed_config)
        
        # Start the feed
        feed.start()
        logger.info("Live data feed started")
        
        # Keep running while market is open
        while is_market_hours():
            time.sleep(60)  # Check every minute
            
            # Log connection status
            if feed.is_connected():
                logger.debug("WebSocket connected")
            else:
                logger.warning("WebSocket disconnected, attempting reconnect...")
                try:
                    feed.stop()
                    time.sleep(5)
                    feed.start()
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
        
        # Stop at market close
        feed.stop()
        logger.info("Market closed, engine stopped")
        
    except Exception as e:
        logger.error(f"Engine error: {e}")
        raise

if __name__ == "__main__":
    if is_market_hours():
        run_engine()
    else:
        logger.info("Outside market hours, skipping engine run")
```

### Step 2: Set Up Scheduled Task

1. Go to the **Tasks** tab in PythonAnywhere dashboard
2. Click **Create a new scheduled task**
3. Enter the command:
   ```
   /home/YOUR_USERNAME/.virtualenvs/mini-simon/bin/python /home/YOUR_USERNAME/mini-simon/background_engine.py
   ```
4. Set the schedule to run **every 5 minutes** during market hours:
   ```
   */5 9-15 * * 1-5
   ```
   (This runs every 5 minutes, hours 9-15, Monday-Friday)

### Step 3: Alternative: Always-On Task (Paid Feature)

If you have a paid PythonAnywhere account:

1. Go to the **Always-on tasks** section
2. Create a new always-on task with the same command
3. This will keep the engine running 24/7

---

## 5. Verification Steps

### Check Web App Health

```bash
# SSH into PythonAnywhere and run:
curl https://yourusername.pythonanywhere.com/api/health
```

Expected response:
```json
{
  "engine_running": true,
  "ws_connected": true,
  "ws_live": true,
  "ws_tick_age_s": 2.5
}
```

### Check Logs

```bash
# View cloud deployment log
cat ~/mini-simon/logs/cloud_deploy.log

# View live engine log
tail -f ~/mini-simon/logs/live_engine.log

# View error log (from Web tab in dashboard)
cat /var/log/www.yourusername.pythonanywhere.com.error.log
```

### Verify IST Timestamps

All timestamps should be in IST (Asia/Kolkata). Check in logs:

```bash
grep "IST" ~/mini-simon/logs/cloud_deploy.log
grep "2024-" ~/mini-simon/logs/live_engine.log | head -5
```

---

## 6. Troubleshooting

### Issue: WebSocket Not Connecting

**Solution:**
1. Check Fyers credentials in `.env`
2. Verify access token is not expired
3. Check `logs/cloud_deploy.log` for errors
4. Ensure WebSocket library is installed: `pip show websocket-client`

### Issue: App Returns 500 Error

**Solution:**
1. Check error log: `/var/log/www.yourusername.pythonanywhere.com.error.log`
2. Verify WSGI file syntax
3. Test locally: `python -c "from web_main import app"`

### Issue: Static Files Not Loading

**Solution:**
1. Check static files directory exists: `ls ~/mini-simon/static/`
2. Verify permissions: `chmod -R 755 ~/mini-simon/static/`
3. Check CSS/JS paths in templates use `/static/` prefix

### Issue: Logs Not Writing

**Solution:**
1. Check directory permissions: `ls -la ~/mini-simon/logs/`
2. Verify path in `cloud_utils.get_logs_path()`
3. Ensure log directory is writable: `chmod 755 ~/mini-simon/logs/`

### Issue: Scheduled Task Not Running

**Solution:**
1. Check task syntax in PythonAnywhere Tasks tab
2. Verify virtual environment path
3. Check task log: `/home/YOUR_USERNAME/task_log.background_engine.log`
4. Test command manually in Bash console

---

## 7. Maintenance Commands

```bash
# Activate virtual environment
workon mini-simon

# Update dependencies
cd ~/mini-simon
pip install --upgrade -r requirements.txt

# Check disk space
df -h

# View running processes
ps aux | grep python

# Restart web app (from Web tab or via API)
# Go to PythonAnywhere Web tab and click Reload

# Backup logs
cd ~/mini-simon/logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz *.log

# Clean old logs (keep last 7 days)
find ~/mini-simon/logs -name "*.log" -mtime +7 -delete
```

---

## 8. Security Checklist

- [ ] `.env` file added to `.gitignore`
- [ ] Fyers credentials NOT committed to Git
- [ ] `credentials.py` ignored or removed from tracking
- [ ] Discord webhook URL kept private
- [ ] Logs directory has proper permissions (755)
- [ ] Virtual environment path is correct
- [ ] WSGI file doesn't expose sensitive data

---

## 9. IST Timezone Verification

All timestamps in Mini Simon are strictly IST (Asia/Kolkata). To verify:

```python
# Run this in PythonAnywhere console:
from time_utils import GlobalTimeHandler
print(GlobalTimeHandler.now_ist_str())

# Should show current time in IST format: "30-Mar-2026 10:53 AM"
```

---

## Support

For issues specific to PythonAnywhere:
- Check [PythonAnywhere Help](https://help.pythonanywhere.com/)
- Review [Web App Setup Guide](https://help.pythonanywhere.com/pages/WebAppSetup/)
- Check [WSGI Configuration](https://help.pythonanywhere.com/pages/WSGIFile/)

For Mini Simon specific issues:
- Review `logs/cloud_deploy.log`
- Check `logs/live_engine.log`
- Verify environment variables are set correctly
