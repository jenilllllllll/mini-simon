# fyers_login.py

from fyers_apiv3 import fyersModel
import credentials as cd

# STEP 1: Load your credentials
client_id = cd.client_id
secret_key = cd.secret_key
redirect_uri = cd.redirect_uri
response_type = "code"
state = "sample_state"
grant_type = "authorization_code"

# STEP 2: Generate the auth URL
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type
)

auth_url = session.generate_authcode()
print("\n🔗 Open this URL in your browser and complete login:")
print(auth_url)

# STEP 3: Ask user to paste redirected URL after login
redirected_url = input("\n📥 Paste the redirected URL after login: ").strip()

# STEP 4: Extract auth_code
try:
    if 'auth_code=' not in redirected_url:
        raise ValueError("Missing 'auth_code=' in the URL.")

    # Safely extract the auth_code
    start = redirected_url.index("auth_code=") + len("auth_code=")
    end = redirected_url.index("&state")
    auth_code = redirected_url[start:end]
    print(f"\n✅ Extracted Auth Code: {auth_code}")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("🔁 Please make sure you pasted the correct URL after logging in.")
    exit(1)

# STEP 5: Exchange auth_code for access_token
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type,
    grant_type=grant_type
)

session.set_token(auth_code)

try:
    token_response = session.generate_token()

    if "access_token" not in token_response:
        raise ValueError("Access token not found in response.")

    access_token = token_response["access_token"]
    print(f"\n✅ Access Token Generated:\n{access_token}")

    # Save to file
    with open("access.txt", "w") as f:
        f.write(access_token)
    print("💾 Access token saved to 'access.txt'.")

except Exception as e:
    print(f"\n❌ Failed to generate access token: {str(e)}")
    exit(1)
