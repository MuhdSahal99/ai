import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from utils.job_utils import check_job_ids
from services.vector_store import vector_store
import logging
from utils.job_utils import check_and_fix_job_ids

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

app.register_blueprint(api_bp, url_prefix='/api')

# logging 
logging.basicConfig(level=logging.INFO)

initialization_has_run = False

def initialize():
    global initialization_has_run
    if not initialization_has_run:
        logging.info("Checking and fixing job IDs...")
        job_ids = check_and_fix_job_ids(vector_store)
        logging.info(f"Available job IDs after initialization: {job_ids}")
        initialization_has_run = True
    
app.before_request(initialize) 

if __name__ == '__main__':
    app.run(debug=True, port=5000)