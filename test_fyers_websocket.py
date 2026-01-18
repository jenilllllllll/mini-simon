"""
Test Fyers WebSocket Connection
Tests different WebSocket endpoints and connection methods
"""

import websocket
import json
import threading
import time
from fyers_apiv3 import fyersModel

class FyersWebSocketTest:
    def __init__(self, app_id, access_token):
        self.app_id = app_id
        self.access_token = access_token
        self.ws = None
        self.connected = False
        
    def on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            print(f"Message received: {data}")
            
            if data.get('type') == 'sf':
                print(f"Tick data: {data.get('symbol')} - LTP: {data.get('ltp')}")
            elif data.get('s') == 'ok':
                print(f"Subscription successful: {data}")
            else:
                print(f"Other message: {data}")
                
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        print("WebSocket connected successfully!")
        self.connected = True
        
        # Send subscription message
        subscribe_msg = {
            "type": "sf",
            "symbols": [
                {"symbol": "NSE:RELIANCE", "type": "sf"},
                {"symbol": "NSE:TCS", "type": "sf"}
            ]
        }
        
        ws.send(json.dumps(subscribe_msg))
        print("Subscription sent")
    
    def test_connection(self, url):
        """Test WebSocket connection with given URL"""
        print(f"\n=== Testing WebSocket URL: {url} ===")
        
        try:
            websocket.enableTrace(True)
            self.ws = websocket.WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection or timeout
            timeout = 10
            for i in range(timeout):
                if self.connected:
                    print("✅ Connection successful!")
                    time.sleep(5)  # Let it run for a few seconds
                    self.ws.close()
                    return True
                time.sleep(1)
            
            print("❌ Connection timeout")
            self.ws.close()
            return False
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

def main():
    app_id = 'CAALOFK6YE-100'
    access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcEo5cVVqNEZIaDBhUlVMT0Y2ajd3eHlTWGs3bzZFWFJ0UEpmUTNXWWdnalI1ZGNwQzIwY2Jka1JDdHl3eTc4RGtFR0FyU3pmdVRucGdMb2lxOTJnRzA0UFl6cmxfRDFsNXVQekgxMDV2SW1PVFUyOD0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI2ZTMxYzhjMmQwMDFiMjQwMWZjN2NkZDdjYTZjNGJiZjdlODRiOGMxMGQ0ZjEwYmZiZGJiNTFiMiIsImlzRGRwaUVuYWJsZWQiOiJZIiwiaXNNdGZFbmFibGVkIjoiWSIsImZ5X2lkIjoiWUowMzUzMyIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzY0Mjg5ODAwLCJpYXQiOjE3NjQyMTk1NDAsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc2NDIxOTU0MCwic3ViIjoiYWNjZXNzX3Rva2VuIn0.EvimQ491NsaDqp5LYNAz-4ju1MJC6MN37lFHu-sBD4k'
    
    tester = FyersWebSocketTest(app_id, access_token)
    
    # Test different WebSocket URLs
    urls = [
        f"wss://ws.fyers.in/v1/data-feed?access_token={access_token}&client_id={app_id}",
        f"wss://ws.fyers.in/api/v2/data-feed?access_token={access_token}&client_id={app_id}",
        f"wss://api.fyers.in/v1/data-feed?access_token={access_token}&client_id={app_id}",
        f"wss://api-t1.fyers.in/data-feed?access_token={access_token}&client_id={app_id}",
        f"wss://fyers.in/data-feed?access_token={access_token}&client_id={app_id}"
    ]
    
    for url in urls:
        success = tester.test_connection(url)
        if success:
            print(f"\n🎉 Found working URL: {url}")
            break
        time.sleep(2)
    
    print("\n🔍 WebSocket testing complete!")

if __name__ == "__main__":
    main()
