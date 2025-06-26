# Import the required module from the fyers_apiv3 package
from fyers_apiv3 import fyersModel

# Replace these values with your actual API credentials
client_id = "CAALOFK6YE-100"
secret_key = "FOXFRTSHB9"
redirect_uri = "https://www.google.com"
response_type = "code"  
state = "sample_state"
user_name = "YJ03533"
pin1 = "5"
pin2 = "3"
pin3 = "5"
pin4 = "4"


totp_key = "GZ3ELFBKXFYJAQGBXEQCHPCJNHLLY245"


# Create a session model with the provided credentials
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type
)

# Generate the auth code using the session model
response = session.generate_authcode()

# Print the auth code received in the response
print(response)

