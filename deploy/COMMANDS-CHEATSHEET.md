# Mini-Simon Deployment Command Cheatsheet

## 🚀 Quick Commands

### Start Everything
```bash
sudo pm2 start ecosystem.config.js && sudo systemctl restart nginx
```

### Stop Everything
```bash
sudo pm2 stop all && sudo systemctl stop nginx
```

### Restart Application
```bash
sudo pm2 restart mini-simon-dashboard
```

### View Status
```bash
sudo pm2 status && sudo systemctl status nginx && sudo ufw status
```

---

## 📊 PM2 Commands

| Command | Description |
|---------|-------------|
| `sudo pm2 start ecosystem.config.js` | Start all processes |
| `sudo pm2 stop mini-simon-dashboard` | Stop specific process |
| `sudo pm2 restart all` | Restart all processes |
| `sudo pm2 delete all` | Remove all from PM2 |
| `sudo pm2 logs` | View all logs in real-time |
| `sudo pm2 logs mini-simon-dashboard --lines 100` | View last 100 lines |
| `sudo pm2 monit` | Interactive monitoring |
| `sudo pm2 flush` | Clear all logs |
| `sudo pm2 save` | Save current config |
| `sudo pm2 startup` | Generate startup script |
| `sudo pm2 describe mini-simon-dashboard` | Detailed process info |

---

## 🌐 Nginx Commands

| Command | Description |
|---------|-------------|
| `sudo nginx -t` | Test configuration |
| `sudo systemctl start nginx` | Start Nginx |
| `sudo systemctl stop nginx` | Stop Nginx |
| `sudo systemctl restart nginx` | Restart Nginx |
| `sudo systemctl reload nginx` | Reload config (zero-downtime) |
| `sudo systemctl status nginx` | View Nginx status |

### Log Files
```bash
sudo tail -f /var/log/nginx/mini-simon-access.log
sudo tail -f /var/log/nginx/mini-simon-error.log
sudo tail -f /var/log/mini-simon/out.log
sudo tail -f /var/log/mini-simon/error.log
```

---

## 🔥 UFW Firewall Commands

| Command | Description |
|---------|-------------|
| `sudo ufw status` | Show firewall status |
| `sudo ufw status verbose` | Detailed status |
| `sudo ufw allow 22/tcp` | Allow SSH |
| `sudo ufw allow 80/tcp` | Allow HTTP |
| `sudo ufw allow 443/tcp` | Allow HTTPS |
| `sudo ufw deny 8000/tcp` | Block direct app access |
| `sudo ufw enable` | Enable firewall |
| `sudo ufw disable` | Disable firewall |
| `sudo ufw reset` | Reset to defaults |

---

## 🐍 Python/Virtual Environment

```bash
# Activate virtual environment
source /opt/mini-simon/venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Check installed packages
pip list

# Deactivate
 deactivate
```

---

## 🔍 Debugging Commands

### Check What's Running
```bash
# All processes
ps aux | grep -E "mini-simon|nginx|pm2"

# Network connections
sudo netstat -tulpn | grep -E "8000|80|443"

# Check ports
sudo lsof -i :8000
sudo lsof -i :80
sudo lsof -i :443
```

### System Resources
```bash
# Memory usage
free -h

# Disk usage
df -h

# CPU and processes
top
htop  # (if installed)

# Bandwidth usage
iftop  # (if installed)
```

### WebSocket Testing
```bash
# Test WebSocket endpoint via curl
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Host: your-server-ip" \
     -H "Origin: http://your-server-ip" \
     http://your-server-ip/ws
```

---

## 📝 Log Analysis

### Search Logs
```bash
# Find errors in last hour
sudo grep "ERROR" /var/log/mini-simon/error.log | tail -20

# Search for specific symbol
sudo grep "NIFTY" /var/log/mini-simon/out.log | tail -50

# Watch live logs
sudo tail -f /var/log/mini-simon/combined.log | grep "WebSocket"
```

### Log Rotation
```bash
# Install PM2 log rotate
sudo pm2 install pm2-logrotate

# Configure
sudo pm2 set pm2-logrotate:max_size 100M
sudo pm2 set pm2-logrotate:retain 10
sudo pm2 set pm2-logrotate:compress true
```

---

## 🔄 Updates & Deployment

### Update Application Code
```bash
cd /opt/mini-simon
sudo git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo pm2 restart mini-simon-dashboard
```

### Full Redeploy
```bash
cd /opt/mini-simon
sudo pm2 stop all
sudo git pull
source venv/bin/activate
pip install -r requirements.txt
sudo pm2 start ecosystem.config.js
sudo pm2 save
```

---

## 🆘 Emergency Recovery

### If Server Becomes Unresponsive
```bash
# SSH in and check
ssh root@YOUR_DROPLET_IP

# Check if process is running
sudo pm2 status

# If PM2 is not responding
sudo pkill -f "mini-simon"
sudo pm2 resurrect  # Restore saved processes

# Restart from scratch
sudo pm2 delete all
sudo pm2 start ecosystem.config.js
sudo pm2 save
```

### Reset Everything
```bash
# WARNING: This stops everything!
sudo pm2 stop all
sudo systemctl stop nginx
sudo ufw disable
# Debug issues, then re-enable
sudo ufw enable
sudo systemctl start nginx
sudo pm2 start all
```

---

## 📈 Performance Tuning

### Increase File Descriptor Limits
```bash
# Check current limits
ulimit -n

# Increase (add to /etc/security/limits.conf)
sudo nano /etc/security/limits.conf
# Add:
# mini-simon soft nofile 65536
# mini-simon hard nofile 65536
```

### Nginx Performance
```bash
# Edit nginx.conf
sudo nano /etc/nginx/nginx.conf

# Add inside http block:
worker_rlimit_nofile 65535;
worker_connections 4096;
```

---

## 🧪 Testing Commands

### Health Checks
```bash
# Test HTTP endpoint
curl -s http://localhost/health || echo "Health check failed"

# Test via Nginx
curl -s http://localhost/api/status

# External test
curl -s http://YOUR_SERVER_IP/
```

### WebSocket Test Script
```bash
# Save as test_ws.sh
#!/bin/bash
wscat -c "ws://YOUR_SERVER_IP/ws" || echo "WebSocket connection failed"
# Install wscat: npm install -g wscat
```

---

## 💾 Backup Commands

```bash
# Backup database
sudo cp /opt/mini-simon/trades.db /opt/mini-simon/backups/trades-$(date +%Y%m%d).db

# Backup logs
sudo tar -czf /opt/mini-simon/backups/logs-$(date +%Y%m%d).tar.gz /var/log/mini-simon/

# Backup config
sudo cp /opt/mini-simon/.env /opt/mini-simon/backups/env-$(date +%Y%m%d)
```

---

**Tip**: Bookmark this file or print it for quick reference during deployment!
