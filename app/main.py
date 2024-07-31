from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask
from flask_cors import CORS
from app.routes.EmoSentinel_routes import emosentinel_router as fastapi_emosentinel_router
#from app.routes.flask_routes import blueprint as flask_blueprint

# FastAPI app
fastapi_app = FastAPI()

# Adding CORS middleware to FastAPI app
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

fastapi_app.include_router(fastapi_emosentinel_router)

# Flask app
#flask_app = Flask(__name__)

# Adding CORS support to Flask app
#CORS(flask_app)

#flask_app.register_blueprint(flask_blueprint)

# Function to run the applications
def run_fastapi():
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

#def run_flask():
    #flask_app.run(host="0.0.0.0", port=5000)
