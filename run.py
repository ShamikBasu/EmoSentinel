import threading
from app.main import run_fastapi#, run_flask

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi)
    #flask_thread = threading.Thread(target=run_flask)

    fastapi_thread.start()
    #flask_thread.start()

    fastapi_thread.join()
    #flask_thread.join()
