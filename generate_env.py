"""
Generate .env file from credentials.py for PythonAnywhere deployment.

Run this script to create your .env file:
    python generate_env.py

Then manually add your Fyers Access Token (see instructions below).
"""

import os

# Read credentials from credentials.py
exec(open('credentials.py').read())

# Generate .env content
env_content = f"""# Mini Simon - Environment Configuration
# Generated from credentials.py for PythonAnywhere deployment

# =============================================================================
# FYERS API CREDENTIALS (Required)
# =============================================================================
FYERS_CLIENT_ID={client_id}
FYERS_APP_ID={client_id}
FYERS_SECRET_KEY={secret_key}

# ACCESS TOKEN - YOU MUST GENERATE THIS (see below)
FYERS_ACCESS_TOKEN=YOUR_ACCESS_TOKEN_HERE

# =============================================================================
# FYERS LOGIN CREDENTIALS
# =============================================================================
FYERS_USERNAME={user_name}
FYERS_TOTP_KEY={totp_key}

# =============================================================================
# PYTHONANYWHERE CONFIG (Update with your username)
# =============================================================================
PYTHONANYWHERE_USERNAME=your_username_here
PYTHONANYWHERE_PROJECT_PATH=/home/your_username_here/mini-simon
PYTHONANYWHERE_DOMAIN=pythonanywhere.com

# =============================================================================
# OTHER CONFIG
# =============================================================================
LOG_LEVEL=INFO
PAPER_TRADING_MODE=true
RISK_PER_TRADE=1.0
ACCOUNT_SIZE=100000
WS_MAX_RECONNECT_ATTEMPTS=10
WS_RECONNECT_DELAY=5
WS_HEARTBEAT_INTERVAL=30
WS_TICK_TIMEOUT=90
ENABLE_MCX=true
"""

# Write to .env file
with open('.env', 'w') as f:
    f.write(env_content)

print("=" * 70)
print("✅ .env file generated successfully!")
print("=" * 70)
print()
print("⚠️  IMPORTANT: You still need to add your FYERS ACCESS TOKEN")
print()
print("How to get your Access Token:")
print("1. Run: python credentials.py")
print("2. It will print an auth code URL")
print("3. Visit that URL, complete 2FA login")
print("4. Copy the auth code from redirect URL")
print("5. Exchange auth code for access token using Fyers API")
print("6. Edit .env file and replace YOUR_ACCESS_TOKEN_HERE with the token")
print()
print("Note: Access tokens expire daily. You'll need to refresh them.")
print("=" * 70)
