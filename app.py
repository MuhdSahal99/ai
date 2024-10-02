import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, send_from_directory
from flask_cors import CORS
from api.routes import api_bp
from utils.job_utils import check_job_ids
from services.vector_store import vector_store
import logging
from utils.job_utils import check_and_fix_job_ids
from dotenv import load_dotenv
import os



app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

app.register_blueprint(api_bp, url_prefix='/api')

# logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
       return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

initialization_has_run = False


def initialize():
    global initialization_has_run
    if not initialization_has_run:
        try:
            logger.info("Checking and fixing job IDs...")
            job_ids = check_and_fix_job_ids(vector_store)
            logger.info(f"Available job IDs after initialization: {job_ids}")

            initialization_has_run = True
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
    
app.before_request(initialize) 

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    # Return a generic error message
    return {"error": "An unexpected error occurred"}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)