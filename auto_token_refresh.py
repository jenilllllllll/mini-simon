"""
Auto Token Refresher for Fyers API
Automates the login flow using Selenium + TOTP

Usage:
    python auto_token_refresh.py

This will automatically:
1. Open browser
2. Login with your credentials + TOTP
3. Get auth code
4. Generate access token
5. Update .env file
6. Optionally update Fly.io secrets

Requirements:
    pip install selenium pyotp webdriver-manager
"""

import os
import sys
import time
import re
from urllib.parse import urlparse, parse_qs

# Import credentials
try:
    import credentials as cd
except ImportError:
    print("❌ credentials.py not found!")
    sys.exit(1)

def generate_token_manually():
    """Fallback: Run the existing login_automation.py flow"""
    print("\n" + "="*60)
    print("MANUAL TOKEN GENERATION")
    print("="*60)
    print("\nI'll help you generate a new access token.")
    print("This token expires in 24 hours.\n")
    
    # Generate auth URL
    from fyers_apiv3 import fyersModel
    
    session = fyersModel.SessionModel(
        client_id=cd.client_id,
        secret_key=cd.secret_key,
        redirect_uri=cd.redirect_uri,
        response_type="code"
    )
    
    auth_url = session.generate_authcode()
    
    print("🔐 STEP 1: Open this URL in your browser:")
    print(f"\n   {auth_url}\n")
    
    print("📋 STEP 2: Login with:")
    print(f"   Username: {cd.user_name}")
    print(f"   PIN: {cd.pin1}{cd.pin2}{cd.pin3}{cd.pin4}")
    
    # Generate TOTP
    try:
        import pyotp
        totp = pyotp.TOTP(cd.totp_key)
        current_totp = totp.now()
        print(f"   TOTP (auto-generated): {current_totp}")
        print(f"   (Or use authenticator app)")
    except ImportError:
        print(f"   TOTP Key: {cd.totp_key[:10]}... (use authenticator app)")
    
    print("\n📥 STEP 3: After login, copy the FULL redirected URL")
    print("   (It will look like: https://www.google.com?auth_code=XXXX&state=...)")
    print()
    
    redirected_url = input("Paste the redirected URL: ").strip()
    
    # Extract auth code
    try:
        if 'auth_code=' not in redirected_url:
            raise ValueError("Missing 'auth_code=' in URL")
        
        parsed = urlparse(redirected_url)
        params = parse_qs(parsed.query)
        auth_code = params.get('auth_code', [None])[0]
        
        if not auth_code:
            raise ValueError("Could not extract auth_code")
        
        print(f"\n✅ Auth Code: {auth_code[:20]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None
    
    # Exchange for token
    print("\n🔑 Generating access token...")
    
    session = fyersModel.SessionModel(
        client_id=cd.client_id,
        secret_key=cd.secret_key,
        redirect_uri=cd.redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )
    
    session.set_token(auth_code)
    
    try:
        response = session.generate_token()
        access_token = response.get("access_token")
        
        if access_token:
            # Save to file
            with open("access.txt", "w") as f:
                f.write(access_token)
            
            print(f"\n✅ SUCCESS!")
            print(f"Access Token: {access_token[:50]}...")
            print("Saved to: access.txt")
            
            # Update .env
            update_env_token(access_token)
            
            return access_token
        else:
            print("❌ No access token in response")
            return None
            
    except Exception as e:
        print(f"\n❌ Failed to generate token: {e}")
        return None

def update_env_token(token):
    """Update .env file with new token"""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print(f"⚠️  {env_path} not found. Creating new file...")
        # Create basic .env structure
        with open(env_path, 'w') as f:
            f.write(f"FYERS_ACCESS_TOKEN={token}\n")
            f.write(f"FYERS_CLIENT_ID={cd.client_id}\n")
            f.write(f"FYERS_APP_ID={cd.client_id}\n")
        return
    
    # Read existing .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update or add token
    token_updated = False
    new_lines = []
    
    for line in lines:
        if line.startswith('FYERS_ACCESS_TOKEN='):
            new_lines.append(f'FYERS_ACCESS_TOKEN={token}\n')
            token_updated = True
        else:
            new_lines.append(line)
    
    if not token_updated:
        new_lines.append(f'FYERS_ACCESS_TOKEN={token}\n')
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated {env_path} with new token")

def auto_login_with_selenium():
    """Fully automated login using Selenium"""
    print("\n" + "="*60)
    print("AUTOMATED LOGIN (Selenium)")
    print("="*60)
    
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        import pyotp
    except ImportError:
        print("❌ Required packages not installed.")
        print("Run: pip install selenium pyotp webdriver-manager")
        return generate_token_manually()
    
    print("🤖 Starting automated login...")
    
    # Generate auth URL
    from fyers_apiv3 import fyersModel
    
    session = fyersModel.SessionModel(
        client_id=cd.client_id,
        secret_key=cd.secret_key,
        redirect_uri=cd.redirect_uri,
        response_type="code"
    )
    
    auth_url = session.generate_authcode()
    
    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    # Uncomment for headless mode (no browser window):
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--no-sandbox")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("✓ Chrome started")
        
        # Navigate to auth URL
        driver.get(auth_url)
        print("✓ Navigated to login page")
        
        wait = WebDriverWait(driver, 30)
        
        # Step 1: Enter username
        username_field = wait.until(EC.presence_of_element_located((By.ID, "fyers_id")))
        username_field.send_keys(cd.user_name)
        print("✓ Username entered")
        
        # Step 2: Enter password (PIN)
        pin = f"{cd.pin1}{cd.pin2}{cd.pin3}{cd.pin4}"
        password_field = driver.find_element(By.ID, "password")
        password_field.send_keys(pin)
        print("✓ PIN entered")
        
        # Step 3: Generate and enter TOTP
        totp = pyotp.TOTP(cd.totp_key)
        totp_code = totp.now()
        
        totp_field = driver.find_element(By.ID, "totp")
        totp_field.send_keys(totp_code)
        print(f"✓ TOTP entered: {totp_code}")
        
        # Step 4: Click login
        login_btn = driver.find_element(By.ID, "btn_login")
        login_btn.click()
        print("✓ Login clicked")
        
        # Wait for redirect
        time.sleep(5)
        
        # Check if we're on the redirect URL
        current_url = driver.current_url
        print(f"\n📍 Current URL: {current_url[:80]}...")
        
        if "auth_code=" in current_url:
            print("✅ Login successful!")
            
            # Extract auth code
            parsed = urlparse(current_url)
            params = parse_qs(parsed.query)
            auth_code = params.get('auth_code', [None])[0]
            
            driver.quit()
            
            # Exchange for token
            print("\n🔑 Exchanging auth code for access token...")
            
            session = fyersModel.SessionModel(
                client_id=cd.client_id,
                secret_key=cd.secret_key,
                redirect_uri=cd.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            
            session.set_token(auth_code)
            response = session.generate_token()
            access_token = response.get("access_token")
            
            if access_token:
                # Save
                with open("access.txt", "w") as f:
                    f.write(access_token)
                
                update_env_token(access_token)
                
                print(f"\n🎉 SUCCESS!")
                print(f"Access Token: {access_token[:50]}...")
                print("\nToken saved to:")
                print("  - access.txt")
                print("  - .env (FYERS_ACCESS_TOKEN)")
                
                return access_token
        else:
            print("⚠️  Not redirected yet. Check browser manually.")
            input("Press Enter after completing login manually...")
            
            current_url = driver.current_url
            driver.quit()
            
            if "auth_code=" in current_url:
                # Extract and continue
                parsed = urlparse(current_url)
                params = parse_qs(parsed.query)
                auth_code = params.get('auth_code', [None])[0]
                
                # Exchange for token
                session = fyersModel.SessionModel(
                    client_id=cd.client_id,
                    secret_key=cd.secret_key,
                    redirect_uri=cd.redirect_uri,
                    response_type="code",
                    grant_type="authorization_code"
                )
                
                session.set_token(auth_code)
                response = session.generate_token()
                access_token = response.get("access_token")
                
                if access_token:
                    with open("access.txt", "w") as f:
                        f.write(access_token)
                    update_env_token(access_token)
                    return access_token
    
    except Exception as e:
        print(f"\n❌ Automation failed: {e}")
        print("Falling back to manual mode...\n")
        return generate_token_manually()

def main():
    print_header()
    
    # Check if token already exists and is recent
    if os.path.exists("access.txt"):
        modified_time = os.path.getmtime("access.txt")
        age_hours = (time.time() - modified_time) / 3600
        
        print(f"Existing token found (age: {age_hours:.1f} hours)")
        
        if age_hours < 20:  # Less than 20 hours old
            print("✅ Token is still valid (expires in 24 hours)")
            
            with open("access.txt", 'r') as f:
                token = f.read().strip()
            
            if token:
                print(f"\nToken: {token[:50]}...")
                
                # Update .env just in case
                update_env_token(token)
                
                choice = input("\nGenerate new token anyway? (y/N): ").lower()
                if choice != 'y':
                    print("\nUsing existing token.")
                    return
    
    print("\nChoose login method:")
    print("1. Automated (Selenium) - Opens browser, auto-fills credentials")
    print("2. Manual - I'll guide you through the steps")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        token = auto_login_with_selenium()
    else:
        token = generate_token_manually()
    
    if token:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Upload the updated .env file to PythonAnywhere")
        print("2. Reload your web app on PythonAnywhere")
        print("3. Your dashboard should work now!")
        print("\nRemember: This token expires in 24 hours.")
        print("Run this script again tomorrow to refresh it.")
        print("\n" + "="*60)
    else:
        print("\n❌ Failed to generate token. Please try again.")

def print_header():
    print("\n" + "="*70)
    print("     FYERS TOKEN REFRESHER - Mini Simon")
    print("="*70)
    print("\nThis tool helps you generate a new Fyers access token.")
    print("The token expires every 24 hours and needs to be refreshed.")
    print("="*70)

if __name__ == "__main__":
    main()
