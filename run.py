import threading
import webbrowser
import uvicorn

URL = "http://127.0.0.1:8000"

if __name__ == "__main__":
    print(f"\nCare Health System -> {URL}  (Ctrl+C to stop)\n")
    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    uvicorn.run("src.agent.main:app", host="127.0.0.1", port=8000, reload=True)