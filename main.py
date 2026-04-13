import threading
import time
import subprocess
import uvicorn
import requests

# Start API
threading.Thread(target=uvicorn.run, kwargs={"app": "src.app.api.main:app"}, daemon=True).start()

# Wait for API
while True:
    try:
        requests.get("http://localhost:8000")
        break
    except requests.ConnectionError:
        time.sleep(0.5)

# Start Streamlit
subprocess.Popen(["python", "-m", "streamlit", "run", "src/app/ui/Chat.py"]).wait()