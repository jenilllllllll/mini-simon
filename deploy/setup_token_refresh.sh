#!/bin/bash
# =============================================================================
# Setup Automated Token Refresh for Mini-Simon
# =============================================================================
# This script sets up a cron job to automatically refresh Fyers tokens daily
# before market opens (6 AM IST).
#
# Usage: sudo ./setup_token_refresh.sh
# =============================================================================

set -e

echo "=========================================="
echo "Token Refresh Automation Setup"
echo "=========================================="

APP_DIR="/opt/mini-simon"
TOKEN_MANAGER="$APP_DIR/deploy/token_manager.py"
VENV_PYTHON="$APP_DIR/venv/bin/python"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Check if token_manager.py exists
if [ ! -f "$TOKEN_MANAGER" ]; then
    echo "❌ token_manager.py not found at $TOKEN_MANAGER"
    exit 1
fi

# Install required dependencies
echo "📦 Installing dependencies..."
source $APP_DIR/venv/bin/activate
pip install -q selenium pyotp webdriver-manager

# Install Chrome (required for headless automation)
echo "🌐 Installing Chrome..."
if ! command -v google-chrome &> /dev/null; then
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - 2>/dev/null || true
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list
    apt-get update -qq
    apt-get install -y -qq google-chrome-stable
    echo "✅ Chrome installed"
else
    echo "✅ Chrome already installed"
fi

# Create log directory
mkdir -p /var/log/mini-simon
touch /var/log/mini-simon/token_manager.log
chown mini-simon:mini-simon /var/log/mini-simon/token_manager.log

# Create cron job
echo "⏰ Setting up cron job..."

# Remove any existing token refresh cron jobs
crontab -u mini-simon -l 2>/dev/null | grep -v "token_manager.py" | crontab -u mini-simon - || true

# Add new cron job - runs at 6:00 AM IST daily (before market opens at 9:15 AM)
# Also runs at 6:00 PM IST (if morning failed)
(
crontab -u mini-simon -l 2>/dev/null || true
echo "# Mini-Simon Token Refresh - Daily at 6:00 AM and 6:00 PM IST"
echo "0 6 * * * cd $APP_DIR && $VENV_PYTHON $TOKEN_MANAGER --refresh-now >> /var/log/mini-simon/token_cron.log 2>&1"
echo "0 18 * * * cd $APP_DIR && $VENV_PYTHON $TOKEN_MANAGER --refresh-now >> /var/log/mini-simon/token_cron.log 2>&1"
) | crontab -u mini-simon -

echo "✅ Cron jobs installed for user 'mini-simon'"

# Display current crontab
echo ""
echo "📋 Current cron jobs:"
crontab -u mini-simon -l | grep -A2 "Mini-Simon" || echo "No jobs found"

# Create systemd service (alternative to cron for more control)
echo ""
echo "🔧 Creating systemd service..."

cat > /etc/systemd/system/mini-simon-token.service << EOF
[Unit]
Description=Mini-Simon Token Manager Daemon
After=network.target

[Service]
Type=simple
User=mini-simon
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
Environment=DISPLAY=:99
ExecStart=$VENV_PYTHON $TOKEN_MANAGER --daemon
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

# Create systemd timer (runs daily at 6 AM)
cat > /etc/systemd/system/mini-simon-token.timer << EOF
[Unit]
Description=Run Mini-Simon Token Refresh Daily at 6 AM IST
Requires=mini-simon-token.service

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Reload systemd
systemctl daemon-reload

echo "✅ Systemd service and timer created"
echo ""
echo "To use systemd timer instead of cron:"
echo "  sudo systemctl enable mini-simon-token.timer"
echo "  sudo systemctl start mini-simon-token.timer"

# Test token refresh once
echo ""
echo "🧪 Testing token refresh (this will open headless browser)..."
read -p "Press Enter to test token refresh now (or Ctrl+C to skip)..." || true

cd $APP_DIR
$VENV_PYTHON $TOKEN_MANAGER --refresh-now

echo ""
echo "=========================================="
echo "✅ Token Refresh Setup Complete!"
echo "=========================================="
echo ""
echo "📅 Schedule:"
echo "  - Cron: Daily at 6:00 AM IST and 6:00 PM IST"
echo "  - Logs: /var/log/mini-simon/token_manager.log"
echo "  - Cron logs: /var/log/mini-simon/token_cron.log"
echo ""
echo "🔧 Manual commands:"
echo "  Test refresh:  sudo -u mini-simon $VENV_PYTHON $TOKEN_MANAGER --refresh-now"
echo "  Check token:    sudo -u mini-simon $VENV_PYTHON $TOKEN_MANAGER --check"
echo "  View logs:      sudo tail -f /var/log/mini-simon/token_manager.log"
echo ""
echo "⚠️  IMPORTANT: Ensure credentials.py exists with correct TOTP key!"
echo ""
