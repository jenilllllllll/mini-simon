# Mini-Simon Token Management Guide

## 🔑 The Problem: Fyers Tokens Expire Every 24 Hours

Fyers API access tokens are valid for **24 hours only**. For a 24/7 production deployment, you need **automated token refresh**.

---

## ✅ Solution: Automated Token Refresh

I've created a complete token management system that automatically refreshes your token **daily at 6 AM IST** (before market opens at 9:15 AM).

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Token Refresh Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  06:00 AM IST (Cron Job)                                     │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────┐                                            │
│  │ token_manager│                                            │
│  │     .py      │                                            │
│  └──────────────┘                                            │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │  Headless   │────▶│  Fyers Login │────▶│  New Token   │ │
│  │   Chrome    │     │  (Auto-TOTP) │     │              │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│                                                  │          │
│                                                  ▼          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Update .env ──▶  Reload PM2  ──▶  Discord Alert     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### Option 1: Automatic Setup (Recommended)

After running the main deployment script, simply run:

```bash
cd /opt/mini-simon/deploy
sudo ./setup_token_refresh.sh
```

This will:
- ✅ Install Chrome + Selenium dependencies
- ✅ Setup cron job (6 AM and 6 PM IST daily)
- ✅ Create systemd service (optional)
- ✅ Test token refresh immediately

### Option 2: Manual Setup

```bash
# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# 2. Install Python packages
cd /opt/mini-simon
source venv/bin/activate
pip install selenium pyotp webdriver-manager

# 3. Add cron job
sudo crontab -u mini-simon -e

# Add these lines:
0 6 * * * cd /opt/mini-simon && /opt/mini-simon/venv/bin/python /opt/mini-simon/deploy/token_manager.py --refresh-now >> /var/log/mini-simon/token_cron.log 2>&1
0 18 * * * cd /opt/mini-simon && /opt/mini-simon/venv/bin/python /opt/mini-simon/deploy/token_manager.py --refresh-now >> /var/log/mini-simon/token_cron.log 2>&1
```

---

## 📋 Prerequisites

### 1. Credentials File (`credentials.py`)

Must exist at `/opt/mini-simon/credentials.py` with:

```python
client_id = "YOUR_CLIENT_ID"
secret_key = "YOUR_SECRET_KEY"
redirect_uri = "https://www.google.com"
user_name = "YOUR_USERNAME"
pin1 = "1"
pin2 = "2"
pin3 = "3"
pin4 = "4"
totp_key = "YOUR_TOTP_SECRET_KEY"  # From Fyers authenticator setup
```

**How to get TOTP key:**
1. Login to Fyers web platform
2. Go to API → Generate API Key
3. When setting up 2FA, save the TOTP secret key (or extract from QR code)
4. Use a tool like https://2fa.live/ to extract from QR if needed

### 2. Environment Variables (Optional)

For Discord notifications on token refresh:

```bash
sudo nano /opt/mini-simon/.env
# Add:
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK
```

---

## 🧪 Testing Token Refresh

### Test Immediately

```bash
cd /opt/mini-simon
sudo -u mini-simon /opt/mini-simon/venv/bin/python deploy/token_manager.py --refresh-now
```

### Check Current Token Validity

```bash
cd /opt/mini-simon
sudo -u mini-simon /opt/mini-simon/venv/bin/python deploy/token_manager.py --check
```

### Send Test Notification

```bash
cd /opt/mini-simon
sudo -u mini-simon /opt/mini-simon/venv/bin/python deploy/token_manager.py --notify-test
```

---

## 📊 Monitoring

### View Token Manager Logs

```bash
# Real-time logs
sudo tail -f /var/log/mini-simon/token_manager.log

# Cron job logs
sudo tail -f /var/log/mini-simon/token_cron.log

# All Mini-Simon logs
sudo tail -f /var/log/mini-simon/*.log
```

### Check Cron Jobs

```bash
sudo crontab -u mini-simon -l
```

### Check Last Refresh Time

```bash
stat /opt/mini-simon/access.txt
```

---

## 🔧 Troubleshooting

### Chrome Not Found

```bash
# Install Chrome
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google.list
sudo apt-get update
sudo apt-get install -y google-chrome-stable
```

### TOTP Code Incorrect

**Problem:** TOTP code expires quickly (30-second window)

**Solution:** 
1. Ensure server timezone is IST: `timedatectl set-timezone Asia/Kolkata`
2. Check system time: `date`
3. Sync time: `sudo apt-get install -y ntp && sudo systemctl restart ntp`

### Token Refresh Fails

**Check logs:**
```bash
sudo tail -100 /var/log/mini-simon/token_manager.log
```

**Common issues:**
- Missing `credentials.py` → Create file with correct credentials
- Wrong TOTP key → Re-setup 2FA on Fyers and save new key
- Chrome crash → Check disk space: `df -h`
- Network issues → Verify internet connection

### Manual Token Refresh (Emergency)

If automation fails, manually refresh:

```bash
# On your local machine with browser:
python auto_token_refresh.py

# Copy the new token to server:
scp access.txt root@YOUR_DROPLET_IP:/opt/mini-simon/

# On server:
cd /opt/mini-simon
sudo pm2 restart all
```

---

## 📅 Refresh Schedule

| Time (IST) | Action | Reason |
|------------|--------|--------|
| 06:00 AM | Auto refresh token | Before market opens (9:15 AM) |
| 06:00 PM | Backup refresh | If morning failed |
| Manual | Run `--refresh-now` | Emergency refresh |

---

## 🔔 Notifications

Set up Discord/Slack webhook for alerts:

```bash
# Edit .env
sudo nano /opt/mini-simon/.env

# Add webhook URL
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/XXXX/YYYY
```

You'll get notifications for:
- ✅ Successful token refresh
- ❌ Failed refresh (requires manual intervention)
- 🧪 Test notifications

---

## 🔒 Security Notes

1. **credentials.py** contains sensitive data - ensure:
   - File permissions: `chmod 600 /opt/mini-simon/credentials.py`
   - Not committed to git (add to `.gitignore`)
   
2. **TOTP Key** is equivalent to your password - keep it secure

3. **Token logs** don't contain full tokens (truncated for security)

---

## 📞 Quick Commands

```bash
# Force immediate refresh
cd /opt/mini-simon && sudo -u mini-simon venv/bin/python deploy/token_manager.py --refresh-now

# Check if token is valid
cd /opt/mini-simon && sudo -u mini-simon venv/bin/python deploy/token_manager.py --check

# View refresh logs
sudo tail -f /var/log/mini-simon/token_manager.log

# Restart application after manual token update
sudo pm2 restart all
```

---

## ✅ Verification Checklist

- [ ] `credentials.py` exists with correct TOTP key
- [ ] Chrome installed: `google-chrome --version`
- [ ] Dependencies installed: `pip list | grep -E "selenium|pyotp"`
- [ ] Cron job active: `sudo crontab -u mini-simon -l`
- [ ] Test refresh successful: `--refresh-now` works
- [ ] Discord webhook configured (optional)
- [ ] Logs writing to `/var/log/mini-simon/`

---

**Result:** 24/7 automated trading with hands-free token management! 🎉
