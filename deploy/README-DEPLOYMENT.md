# Mini-Simon Production Deployment Guide

Complete deployment documentation for hosting Mini-Simon on DigitalOcean Ubuntu Droplet with 24/7 uptime guarantee.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Step-by-Step Manual Setup](#step-by-step-manual-setup)
5. [WebSocket Configuration](#websocket-configuration)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DigitalOcean Droplet                    │
│                      Ubuntu 22.04/24.04 LTS                  │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │   Nginx      │───▶│   PM2        │───▶│  Mini-Simon    │ │
│  │  (Port 80)   │    │  (Process    │    │  (Port 8000)   │ │
│  │  Reverse     │    │   Manager)   │    │  FastAPI App   │ │
│  │   Proxy      │    │              │    │                │ │
│  └──────────────┘    └──────────────┘    └────────────────┘ │
│         │                                              │      │
│         │         WebSocket Support                   │      │
│         │         (Upgrade/Connection Headers)       │      │
│         └─────────────────────────────────────────────┘      │
│                                                              │
│  UFW Firewall: 22 (SSH), 80 (HTTP), 443 (HTTPS)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

- **Server**: DigitalOcean Ubuntu 22.04/24.04 LTS Droplet (2GB+ RAM recommended)
- **Domain** (optional): Point A record to Droplet IP
- **Credentials**: Fyers API credentials (APP_ID, ACCESS_TOKEN)

---

## 🚀 Quick Start

### 1. Create Droplet
```bash
# SSH into your new Droplet
ssh root@YOUR_DROPLET_IP
```

### 2. Clone and Deploy
```bash
# Clone repository (replace with your repo)
git clone https://github.com/yourusername/mini-simon.git /opt/mini-simon

# Navigate to deploy directory
cd /opt/mini-simon/deploy

# Run deployment script
chmod +x launch.sh
sudo ./launch.sh
```

### 3. Configure Environment
```bash
# Edit environment file
sudo nano /opt/mini-simon/.env
```

Add your Fyers credentials:
```env
FYERS_APP_ID=YOUR_APP_ID
FYERS_ACCESS_TOKEN=YOUR_ACCESS_TOKEN
DISCORD_WEBHOOK_URL=optional_webhook_url
```

### 4. Start Application
```bash
cd /opt/mini-simon
sudo pm2 start ecosystem.config.js
sudo pm2 save
sudo pm2 startup
```

### 5. Verify Deployment
```bash
# Check all services
sudo systemctl status nginx
sudo pm2 status
sudo ufw status

# Access dashboard
# Open http://YOUR_DROPLET_IP in browser
```

---

## 🔧 Step-by-Step Manual Setup

### Step 1: Environment Setup

```bash
# Set timezone to IST (Asia/Kolkata)
sudo timedatectl set-timezone Asia/Kolkata

# Update packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Install Node.js & PM2
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

# Install Nginx
sudo apt-get install -y nginx
```

### Step 2: Virtual Environment

```bash
# Create application directory
sudo mkdir -p /opt/mini-simon
sudo useradd --system --no-create-home mini-simon || true

# Create virtual environment
cd /opt/mini-simon
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: PM2 Configuration

The `ecosystem.config.js` file provides:
- **Auto-restart**: Process restarts on crash
- **Memory limit**: Auto-restart at 2GB RAM usage
- **Log rotation**: Structured logging to `/var/log/mini-simon/`
- **Startup script**: Auto-start on server reboot

```bash
# Copy configuration
sudo cp deploy/ecosystem.config.js /opt/mini-simon/

# Start with PM2
cd /opt/mini-simon
sudo pm2 start ecosystem.config.js

# Save PM2 config (auto-start on boot)
sudo pm2 save
sudo pm2 startup systemd
```

### Step 4: Nginx Configuration

**Critical for WebSocket**: The Nginx config includes the essential `Upgrade` and `Connection` headers:

```nginx
# WebSocket endpoint - CRITICAL headers
location /ws {
    proxy_pass http://mini_simon_backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400s;
    proxy_buffering off;
}
```

Install:
```bash
sudo cp deploy/nginx-mini-simon.conf /etc/nginx/sites-available/mini-simon
sudo ln -sf /etc/nginx/sites-available/mini-simon /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### Step 5: Firewall (UFW)

```bash
# Reset UFW
sudo ufw --force reset

# Configure
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# Enable
sudo ufw --force enable
sudo ufw status verbose
```

---

## 🔌 WebSocket Configuration

### Cloud Optimizations in `websocket_manager.py`

The WebSocket manager is already optimized for cloud deployment:

| Feature | Configuration | Purpose |
|---------|--------------|---------|
| Reconnection | Exponential backoff (2s → 5s → 10s) | Handles network interruptions |
| Heartbeat | 10-second timeout | Detects silent disconnections |
| SSL Patch | Applied automatically | Ensures TLS compatibility |
| Threading | Daemon threads | Prevents zombie processes |

### Nginx WebSocket Headers

**Required headers for WebSocket passthrough**:
```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_read_timeout 86400s;
proxy_buffering off;
```

---

## 📊 Monitoring & Maintenance

### PM2 Commands
```bash
# View real-time logs
sudo pm2 logs mini-simon-dashboard

# Monitor CPU/Memory
sudo pm2 monit

# Restart application
sudo pm2 restart mini-simon-dashboard

# View status
sudo pm2 status

# Flush logs
sudo pm2 flush
```

### Nginx Commands
```bash
# Test configuration
sudo nginx -t

# Reload (zero-downtime)
sudo systemctl reload nginx

# View access logs
sudo tail -f /var/log/nginx/mini-simon-access.log

# View error logs
sudo tail -f /var/log/nginx/mini-simon-error.log
```

### System Monitoring
```bash
# Disk usage
df -h

# Memory usage
free -h

# Running processes
ps aux | grep mini-simon

# Network connections
netstat -tulpn | grep :8000
```

---

## 🐛 Troubleshooting

### WebSocket Connection Drops

**Symptom**: Dashboard shows "Disconnected" or live data stops

**Solutions**:
1. Check Nginx error logs:
   ```bash
   sudo tail -f /var/log/nginx/mini-simon-error.log
   ```

2. Verify WebSocket headers in Nginx config

3. Check application logs:
   ```bash
   sudo pm2 logs
   ```

4. Restart WebSocket connection:
   ```bash
   sudo pm2 restart mini-simon-dashboard
   ```

### 502 Bad Gateway

**Symptom**: Nginx returns 502 error

**Solutions**:
1. Verify application is running:
   ```bash
   sudo pm2 status
   sudo netstat -tulpn | grep 8000
   ```

2. Check application logs for startup errors

3. Restart services:
   ```bash
   sudo pm2 restart all
   sudo systemctl restart nginx
   ```

### High Memory Usage

**Symptom**: PM2 auto-restarts due to memory limit

**Solutions**:
1. Check memory usage:
   ```bash
   sudo pm2 monit
   ```

2. Review log file sizes:
   ```bash
   sudo du -sh /var/log/mini-simon/
   ```

3. Configure log rotation:
   ```bash
   sudo pm2 install pm2-logrotate
   sudo pm2 set pm2-logrotate:max_size 100M
   sudo pm2 set pm2-logrotate:retain 10
   ```

### Timezone Issues

**Symptom**: Market hours not aligned with IST

**Solution**:
```bash
sudo timedatectl set-timezone Asia/Kolkata
timedatectl  # Verify
```

---

## 🔒 Security Checklist

- [x] UFW firewall active (only ports 22, 80, 443 open)
- [x] Application runs as non-root user (`mini-simon`)
- [x] Nginx reverse proxy hides internal port 8000
- [x] Sensitive files (`.env`, `.log`, `.db`) blocked by Nginx
- [x] Security headers enabled (X-Frame-Options, X-XSS-Protection)

---

## 📞 Support Commands Reference

```bash
# Full restart sequence
sudo pm2 restart mini-simon-dashboard && sudo systemctl reload nginx

# View all logs
sudo pm2 logs
sudo tail -f /var/log/nginx/mini-simon-error.log

# Check system health
sudo systemctl status nginx
sudo pm2 status
sudo ufw status
```

---

**Last Updated**: May 2026  
**Compatible With**: Ubuntu 22.04/24.04 LTS, Python 3.11, Node.js 20.x
