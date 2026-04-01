# Mini Simon - Fly.io Deployment Guide (24/7 Live Service)

## Why Fly.io?
Fly.io provides **true 24/7 service** with always-on machines, unlike PythonAnywhere's limited background task execution.

## Prerequisites

1. **Install Fly.io CLI**:
```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Mac/Linux
curl -L https://fly.io/install.sh | sh
```

2. **Sign up / Login**:
```bash
fly auth signup   # If you don't have an account
fly auth login    # If you already have an account
```

## Deployment Steps

### Step 1: Push Your Code to GitHub
```bash
git add .
git commit -m "Fly.io deployment ready"
git push origin main
```

### Step 2: Create Fly.io App
```bash
cd "c:\Drive data\Bed Room Trader\mini-simon"
fly launch --name mini-simon --region bom --no-deploy
```
**Note**: Choose `bom` (Mumbai) region for best latency to Indian markets (NSE/MCX).

### Step 3: Set Environment Secrets
```bash
# Set your Fyers credentials (these are encrypted and secure)
fly secrets set FYERS_APP_ID=CAALOFK6YE-100
fly secrets set FYERS_ACCESS_TOKEN=YOUR_ACCESS_TOKEN_HERE
fly secrets set FYERS_CLIENT_ID=CAALOFK6YE-100
fly secrets set PAPER_TRADING_MODE=true
```

### Step 4: Deploy
```bash
fly deploy
```

### Step 5: Verify
```bash
# Check app status
fly status

# View logs
fly logs

# Open in browser
fly open
```

Your app will be available at: `https://mini-simon.fly.dev`

## Managing the App

### View Logs
```bash
fly logs
```

### Restart App
```bash
fly restart
```

### Scale (if needed)
```bash
# Scale to more memory
fly scale memory 512

# Scale to dedicated CPU
fly scale vm dedicated-cpus=1
```

### Update Code
```bash
# After making changes locally
git push origin main
fly deploy
```

## Token Refresh (Every 24 Hours)

Since Fyers tokens expire every 24 hours, you need to refresh them:

### Option 1: Update Secrets (Quick)
```bash
# Generate new token locally first
python auto_token_refresh.py

# Then update Fly.io secrets
fly secrets set FYERS_ACCESS_TOKEN=NEW_TOKEN_HERE
fly restart
```

### Option 2: SSH into Machine
```bash
fly ssh console
python auto_token_refresh.py
```

## Costs

Fly.io free tier includes:
- 3 shared-cpu-1x 256mb VMs (good for testing)
- 3GB persistent volume storage
- 160GB outbound data transfer

For 24/7 operation, you may need to add a payment method, but costs are minimal (~$2-5/month for basic usage).

## Troubleshooting

### App Won't Start
```bash
fly logs --app mini-simon
```

### Check Environment Variables
```bash
fly ssh console
echo $FYERS_APP_ID
echo $FYERS_ACCESS_TOKEN
```

### Rebuild from Scratch
```bash
fly destroy mini-simon
fly launch --name mini-simon --region bom
```

## Health Check Endpoint

Your app includes a health check at:
```
https://mini-simon.fly.dev/api/health
```

Fly.io will automatically restart the app if health checks fail.

## Important Notes

1. **Token Expiry**: Remember to refresh your Fyers token every 24 hours
2. **Paper Trading**: Default is `true` - safe to test
3. **Auto-stop disabled**: `auto_stop_machines = false` in fly.toml keeps it running 24/7
4. **Health checks**: Configured to ensure service availability

## Need Help?

- Fly.io Docs: https://fly.io/docs/
- Fly.io Community: https://community.fly.io/
- Mini Simon Issues: https://github.com/jenilllllllll/mini-simon/issues
