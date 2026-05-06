#!/bin/bash
# =============================================================================
# Mini-Simon Production Deployment Script for DigitalOcean Ubuntu Droplet
# =============================================================================
# This script:
# 1. Updates system packages
# 2. Installs all dependencies (Python, Nginx, PM2, Node.js)
# 3. Sets timezone to IST (Asia/Kolkata)
# 4. Creates service user and directory structure
# 5. Sets up Python virtual environment
# 6. Configures Nginx reverse proxy with WebSocket support
# 7. Configures PM2 process management
# 8. Sets up UFW firewall
# 9. Starts the trading terminal
#
# Usage:
#   chmod +x launch.sh
#   sudo ./launch.sh
#
# Requirements:
#   - Ubuntu 20.04/22.04/24.04 LTS
#   - Root or sudo access
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="mini-simon"
APP_USER="mini-simon"
APP_DIR="/opt/${APP_NAME}"
LOG_DIR="/var/log/${APP_NAME}"
PYTHON_VERSION="3.11"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  Mini-Simon Trading Terminal - Production Deployment${NC}"
echo -e "${BLUE}======================================================${NC}"

# =============================================================================
# Step 1: System Update & Essential Packages
# =============================================================================
echo -e "\n${YELLOW}[1/10] Updating system packages...${NC}"
apt-get update && apt-get upgrade -y
apt-get install -y software-properties-common curl wget git vim htop tree

# =============================================================================
# Step 2: Set Timezone to IST (Asia/Kolkata)
# =============================================================================
echo -e "\n${YELLOW}[2/10] Setting timezone to Indian Standard Time (IST)...${NC}"
timedatectl set-timezone Asia/Kolkata
echo -e "${GREEN}Timezone set to: $(timedatectl | grep "Time zone")${NC}"

# =============================================================================
# Step 3: Install Python & Dependencies
# =============================================================================
echo -e "\n${YELLOW}[3/10] Installing Python ${PYTHON_VERSION} and dependencies...${NC}"
add-apt-repository ppa:deadsnakes/ppa -y || true
apt-get update
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip
apt-get install -y build-essential libssl-dev libffi-dev

# Set Python 3.11 as default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# =============================================================================
# Step 4: Install Node.js & PM2
# =============================================================================
echo -e "\n${YELLOW}[4/10] Installing Node.js and PM2...${NC}"
# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Install PM2 globally
npm install -g pm2

# Install Nginx
echo -e "\n${YELLOW}[4.5/10] Installing Nginx...${NC}"
apt-get install -y nginx

# =============================================================================
# Step 5: Create Application User & Directory Structure
# =============================================================================
echo -e "\n${YELLOW}[5/10] Creating application user and directories...${NC}"

# Create service user (no login, no password)
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /bin/false "$APP_USER"
    echo -e "${GREEN}Created user: $APP_USER${NC}"
else
    echo -e "${YELLOW}User $APP_USER already exists${NC}"
fi

# Create directory structure
mkdir -p "$APP_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/signals"
mkdir -p "$APP_DIR/trades"

# Set ownership
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chown -R "$APP_USER:$APP_USER" "$LOG_DIR"
chmod 755 "$APP_DIR"
chmod 755 "$LOG_DIR"

echo -e "${GREEN}Directory structure created at: $APP_DIR${NC}"

# =============================================================================
# Step 6: Setup Python Virtual Environment
# =============================================================================
echo -e "\n${YELLOW}[6/10] Setting up Python virtual environment...${NC}"

cd "$APP_DIR"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

echo -e "${GREEN}Virtual environment created at: $APP_DIR/venv${NC}"

# =============================================================================
# Step 7: Install Python Dependencies
# =============================================================================
echo -e "\n${YELLOW}[7/10] Installing Python dependencies...${NC}"

# Check if requirements.txt exists in the deploy directory or project root
if [ -f "$APP_DIR/deploy/requirements.txt" ]; then
    pip install -r "$APP_DIR/deploy/requirements.txt"
elif [ -f "$APP_DIR/requirements.txt" ]; then
    pip install -r "$APP_DIR/requirements.txt"
else
    echo -e "${YELLOW}No requirements.txt found. Installing core dependencies...${NC}"
    # Core dependencies for Mini-Simon
    pip install \
        pandas>=1.5.0 \
        numpy>=1.24.0 \
        fyers-apiv3>=3.0.0 \
        fastapi>=0.110.0 \
        "uvicorn[standard]">=0.27.0 \
        websockets>=12.0 \
        websocket-client>=1.6.1 \
        python-dotenv>=1.0.0 \
        pytz>=2024.1 \
        requests>=2.31.0 \
        aiohttp>=3.8.4 \
        psutil>=5.9.0 \
        cryptography>=42.0.0 \
        jinja2>=3.1.0 \
        pydantic>=2.0.0 \
        openpyxl>=3.0.0 \
        pyyaml>=6.0.0
fi

echo -e "${GREEN}Python dependencies installed successfully${NC}"

# =============================================================================
# Step 8: Configure Nginx
# =============================================================================
echo -e "\n${YELLOW}[8/10] Configuring Nginx...${NC}"

# Backup default config
mv /etc/nginx/sites-available/default /etc/nginx/sites-available/default.bak 2>/dev/null || true
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true

# Copy our configuration
if [ -f "$APP_DIR/deploy/nginx-mini-simon.conf" ]; then
    cp "$APP_DIR/deploy/nginx-mini-simon.conf" /etc/nginx/sites-available/mini-simon
else
    # Create Nginx config inline
    cat > /etc/nginx/sites-available/mini-simon << 'EOF'
upstream mini_simon_backend {
    server 127.0.0.1:8000;
    keepalive 64;
}

limit_req_zone $binary_remote_addr zone=api_limit:10m rate=50r/s;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    
    access_log /var/log/nginx/mini-simon-access.log;
    error_log /var/log/nginx/mini-simon-error.log;
    
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    location /static/ {
        alias /opt/mini-simon/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # WebSocket endpoint - CRITICAL headers
    location /ws {
        proxy_pass http://mini_simon_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_buffering off;
        proxy_cache off;
    }
    
    location / {
        proxy_pass http://mini_simon_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        limit_req zone=api_limit burst=100 nodelay;
    }
    
    location ~ /\. {
        deny all;
    }
}
EOF
fi

# Enable site
ln -sf /etc/nginx/sites-available/mini-simon /etc/nginx/sites-enabled/

# Test configuration
nginx -t

# Start/restart Nginx
systemctl enable nginx
systemctl restart nginx

echo -e "${GREEN}Nginx configured and restarted${NC}"

# =============================================================================
# Step 9: Configure PM2
# =============================================================================
echo -e "\n${YELLOW}[9/10] Configuring PM2...${NC}"

# Copy PM2 ecosystem file
if [ -f "$APP_DIR/deploy/ecosystem.config.js" ]; then
    cp "$APP_DIR/deploy/ecosystem.config.js" "$APP_DIR/"
fi

# Create startup script for PM2
env PATH=$PATH:/usr/bin pm2 startup systemd -u "$APP_USER" --hp "/home/$APP_USER" 2>/dev/null || pm2 startup

echo -e "${GREEN}PM2 configured${NC}"

# =============================================================================
# Step 10: Configure UFW Firewall
# =============================================================================
echo -e "\n${YELLOW}[10/10] Configuring UFW Firewall...${NC}"

# Reset and configure UFW
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (critical - don't lock yourself out!)
ufw allow 22/tcp comment 'SSH'

# Allow HTTP/HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Enable firewall
ufw --force enable

# Show status
echo -e "${GREEN}UFW Firewall Status:${NC}"
ufw status verbose

# =============================================================================
# Setup Complete - Summary
# =============================================================================
echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "\n${BLUE}Next Steps:${NC}"
echo -e "  1. Copy your project files to: ${YELLOW}$APP_DIR${NC}"
echo -e "  2. Set environment variables:"
echo -e "     ${YELLOW}sudo nano /opt/mini-simon/.env${NC}"
echo -e "  3. Start the application:"
echo -e "     ${YELLOW}cd $APP_DIR && sudo pm2 start ecosystem.config.js${NC}"
echo -e "  4. Save PM2 config:"
echo -e "     ${YELLOW}sudo pm2 save${NC}"
echo -e "  5. Access the dashboard at:"
echo -e "     ${YELLOW}http://$(curl -s ifconfig.me || echo 'YOUR_SERVER_IP')${NC}"
echo -e "\n${BLUE}Useful Commands:${NC}"
echo -e "  - View logs:    ${YELLOW}sudo pm2 logs${NC}"
echo -e "  - Restart app:  ${YELLOW}sudo pm2 restart mini-simon-dashboard${NC}"
echo -e "  - Check status: ${YELLOW}sudo pm2 status${NC}"
echo -e "  - Nginx test:   ${YELLOW}sudo nginx -t${NC}"
echo -e "  - Nginx reload: ${YELLOW}sudo systemctl reload nginx${NC}"
echo -e "\n${BLUE}Security Notes:${NC}"
echo -e "  - UFW is active and blocking all ports except 22, 80, 443"
echo -e "  - Application runs as user: $APP_USER"
echo -e "  - Logs stored at: $LOG_DIR"
echo -e "${GREEN}======================================================${NC}"
