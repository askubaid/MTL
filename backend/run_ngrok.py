import os
import subprocess
import time
import sys
from pyngrok import ngrok

def main():
    # Start the FastAPI backend
    print("Starting FastAPI backend...")
    # Use sys.executable to ensure uvicorn runs in the same environment
    backend_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    time.sleep(3) # Wait for backend to start
    
    auth_token = os.environ.get("NGROK_AUTHTOKEN")
    if not auth_token:
        print("\n--- NGROK AUTHTOKEN REQUIRED ---")
        print("Please sign up at https://dashboard.ngrok.com/signup if you don't have an account.")
        print("Once signed in, get your authtoken at https://dashboard.ngrok.com/get-started/your-authtoken")
        auth_token = input("Enter your ngrok authtoken (or press Enter if already configured globally): ").strip()
        if auth_token:
            ngrok.set_auth_token(auth_token)
    else:
        ngrok.set_auth_token(auth_token)
    
    try:
        # Open a HTTP tunnel on the default port 8000
        public_url = ngrok.connect(8000)
        print("\n" + "="*60)
        print(f"✅ Backend is running and exposed at: {public_url.public_url}")
        print("👉 Copy the URL above and paste it into the frontend's 'Backend URL' field.")
        print("="*60 + "\n")
        
        # Block until CTRL-C or some other terminating event
        backend_process.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Failed to start ngrok: {e}")
    finally:
        ngrok.kill()
        backend_process.terminate()

if __name__ == '__main__':
    main()
