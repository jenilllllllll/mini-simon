"""
One-Click PythonAnywhere Setup Script for Mini Simon

This script automates the entire setup process. Just run:
    python setup_pythonanywhere.py

And follow the prompts!
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(step_num, text):
    print(f"\n[Step {step_num}] {text}")
    print("-" * 50)

def get_input(prompt, default=None):
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()

def read_credentials():
    """Read credentials from credentials.py"""
    creds = {}
    try:
        with open('credentials.py', 'r') as f:
            content = f.read()
            
        # Extract values
        for line in content.split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                creds[key] = val
    except Exception as e:
        print(f"Error reading credentials.py: {e}")
    return creds

def generate_env_file():
    """Step 1: Generate .env file from credentials"""
    print_step(1, "Generating .env file from your credentials.py")
    
    creds = read_credentials()
    
    if not creds:
        print("❌ Could not read credentials.py. Make sure it exists.")
        return False
    
    print("✓ Found credentials.py")
    print(f"  Client ID: {creds.get('client_id', 'NOT FOUND')}")
    print(f"  Username: {creds.get('user_name', 'NOT FOUND')}")
    
    # Get PythonAnywhere username
    pa_username = get_input("Enter your PythonAnywhere username")
    
    # Get access token
    print("\n⚠️  You need a Fyers Access Token (different from auth code)")
    print("   Options:")
    print("   1. I already have an access token")
    print("   2. Help me generate one")
    
    choice = get_input("Choose (1 or 2)", "1")
    
    access_token = ""
    if choice == "1":
        access_token = get_input("Paste your Fyers Access Token")
    else:
        print("\nTo generate access token:")
        print("  1. Run: python credentials.py")
        print("  2. Visit the URL it prints")
        print("  3. Login and copy the auth code from the redirect URL")
        print("  4. Exchange auth code for access token via Fyers API")
        print("\nFor now, we'll use a placeholder. You can edit .env later.")
        access_token = "PASTE_YOUR_ACCESS_TOKEN_HERE"
    
    # Generate .env content
    env_content = f"""# Mini Simon - Environment Configuration
# Auto-generated from credentials.py

# FYERS API CREDENTIALS
FYERS_CLIENT_ID={creds.get('client_id', '')}
FYERS_APP_ID={creds.get('client_id', '')}
FYERS_ACCESS_TOKEN={access_token}
FYERS_SECRET_KEY={creds.get('secret_key', '')}
FYERS_USERNAME={creds.get('user_name', '')}
FYERS_TOTP_KEY={creds.get('totp_key', '')}

# PYTHONANYWHERE CONFIG
PYTHONANYWHERE_USERNAME={pa_username}
PYTHONANYWHERE_PROJECT_PATH=/home/{pa_username}/mini-simon
PYTHONANYWHERE_DOMAIN=pythonanywhere.com

# OTHER SETTINGS
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
    
    # Write .env file
    env_path = Path('.env')
    env_path.write_text(env_content)
    print(f"\n✓ Created .env file at: {env_path.absolute()}")
    
    if access_token.startswith("PASTE_"):
        print("\n⚠️  IMPORTANT: Edit .env and replace PASTE_YOUR_ACCESS_TOKEN_HERE")
        print("   with your actual Fyers access token!")
    
    return True

def check_requirements():
    """Step 2: Check if requirements.txt is ready"""
    print_step(2, "Checking requirements.txt")
    
    if Path('requirements.txt').exists():
        print("✓ requirements.txt exists")
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'python-dotenv' in content:
                print("✓ python-dotenv included (for .env file support)")
            else:
                print("⚠️  Adding python-dotenv to requirements.txt")
                with open('requirements.txt', 'a') as f:
                    f.write("\npython-dotenv>=1.0.0\n")
    else:
        print("❌ requirements.txt not found!")
        return False
    
    return True

def create_directories():
    """Step 3: Create required directories"""
    print_step(3, "Creating required directories")
    
    dirs = ['logs', 'signals', 'trades', 'Data']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"✓ {d}/")
    
    return True

def update_wsgi():
    """Step 4: Update wsgi.py with correct username"""
    print_step(4, "Checking wsgi.py configuration")
    
    if not Path('wsgi.py').exists():
        print("❌ wsgi.py not found! Run this script from the project root.")
        return False
    
    # Read current content
    with open('wsgi.py', 'r') as f:
        content = f.read()
    
    print("✓ wsgi.py exists")
    print("⚠️  Remember to replace 'YOUR_USERNAME' with your actual username in wsgi.py")
    
    return True

def create_upload_script(pa_username):
    """Create a helper script for uploading to PythonAnywhere"""
    print_step(5, "Creating upload helper script")
    
    upload_script = f'''#!/bin/bash
# PythonAnywhere Upload & Setup Script
# Run this on PythonAnywhere after uploading files

echo "Setting up Mini Simon on PythonAnywhere..."

# 1. Create virtual environment
echo "Creating virtual environment..."
mkvirtualenv mini-simon --python=/usr/bin/python3.10

# 2. Activate and install dependencies
echo "Installing dependencies..."
workon mini-simon
pip install --upgrade pip
pip install -r /home/{pa_username}/mini-simon/requirements.txt

# 3. Create directories
echo "Creating directories..."
cd /home/{pa_username}/mini-simon
mkdir -p logs signals trades Data

# 4. Set permissions
echo "Setting permissions..."
chmod -R 755 logs signals trades Data

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Go to Web tab in PythonAnywhere dashboard"
echo "2. Configure WSGI file (see DEPLOY_PYTHONANYWHERE.md)"
echo "3. Set environment variables in Web tab"
echo "4. Reload the web app"
echo ""
'''
    
    script_path = Path('setup_on_pythonanywhere.sh')
    script_path.write_text(upload_script)
    print(f"✓ Created {script_path}")
    
    return True

def create_instructions(pa_username):
    """Create final instructions file"""
    print_step(6, "Creating upload instructions")
    
    instructions = f'''
╔══════════════════════════════════════════════════════════════════════╗
║           UPLOAD TO PYTHONANYWHERE - FINAL STEPS                     ║
╚══════════════════════════════════════════════════════════════════════╝

Your username: {pa_username}

STEP 1: UPLOAD FILES
═══════════════════
Option A - Using GitHub:
  1. Push your code to GitHub
  2. On PythonAnywhere Bash console:
     cd ~
     git clone https://github.com/YOUR_USERNAME/mini-simon.git

Option B - Manual Upload:
  1. Go to PythonAnywhere Files tab
  2. Create folder: /home/{pa_username}/mini-simon/
  3. Upload ALL these files:
     - wsgi.py
     - web_main.py
     - cloud_utils.py
     - config.py
     - .env (IMPORTANT!)
     - requirements.txt
     - All other .py files
     - templates/ folder
     - static/ folder

STEP 2: RUN SETUP ON PYTHONANYWHERE
═══════════════════════════════════
  1. Open PythonAnywhere Bash console
  2. Run: bash /home/{pa_username}/mini-simon/setup_on_pythonanywhere.sh

STEP 3: CONFIGURE WEB APP
═════════════════════════
  1. Go to Web tab
  2. Click "Add a new web app"
  3. Choose "Manual configuration" → "Python 3.10"
  4. Edit WSGI configuration file:
     - Replace "YOUR_USERNAME" with "{pa_username}"
  5. Set Virtualenv: /home/{pa_username}/.virtualenvs/mini-simon
  6. Add Environment Variables:
     FYERS_APP_ID = your_app_id
     FYERS_ACCESS_TOKEN = your_token
     PYTHONANYWHERE_DOMAIN = pythonanywhere.com
  7. Click RELOAD

STEP 4: VISIT YOUR SITE
═══════════════════════
  https://{pa_username}.pythonanywhere.com

NEED HELP?
══════════
Check DEPLOY_PYTHONANYWHERE.md for detailed instructions.
'''
    
    Path('UPLOAD_INSTRUCTIONS.txt').write_text(instructions)
    print(f"✓ Created UPLOAD_INSTRUCTIONS.txt")
    
    return True

def main():
    print_header("Mini Simon - PythonAnywhere Setup Wizard")
    print("This script will prepare everything for PythonAnywhere deployment.")
    print("Just follow the prompts!\n")
    
    # Check we're in the right directory
    if not Path('credentials.py').exists():
        print("❌ ERROR: Run this script from your mini-simon project folder!")
        print("   Make sure credentials.py is in the same folder.")
        sys.exit(1)
    
    success = True
    
    # Run all steps
    success = generate_env_file() and success
    success = check_requirements() and success
    success = create_directories() and success
    success = update_wsgi() and success
    
    # Get username for remaining steps
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('PYTHONANYWHERE_USERNAME='):
                pa_username = line.split('=', 1)[1].strip()
                break
    else:
        pa_username = "your_username"
    
    create_upload_script(pa_username)
    create_instructions(pa_username)
    
    # Final summary
    print_header("SETUP COMPLETE!")
    
    print("✓ .env file created with your credentials")
    print("✓ All directories created (logs, signals, trades, Data)")
    print("✓ Upload script ready (setup_on_pythonanywhere.sh)")
    print("✓ Instructions written (UPLOAD_INSTRUCTIONS.txt)")
    
    print("\n" + "=" * 70)
    print("NEXT: Follow UPLOAD_INSTRUCTIONS.txt")
    print("=" * 70)
    
    # Read and display the first few lines of instructions
    print("\nPreview of upload steps:")
    print("-" * 50)
    with open('UPLOAD_INSTRUCTIONS.txt', 'r') as f:
        for i, line in enumerate(f):
            if i < 20:
                print(line.rstrip())
    
    print("\n" + "=" * 70)
    print("Questions? Check DEPLOY_PYTHONANYWHERE.md for details.")
    print("=" * 70)

if __name__ == "__main__":
    main()
