from flask import Blueprint,jsonify, request
from .upload import upload_resume, upload_job_description, upload_interview_script
from services.vector_store import vector_store
from models.resume import Resume
from services.job_matching_service import get_job_matches
from services.mistral_service import mistral_service
import logging

api_bp = Blueprint('api', __name__)

api_bp.route('/upload', methods=['POST'])(upload_resume)
api_bp.route('/upload_job_description', methods=['POST'])(upload_job_description)
api_bp.route('/upload_interview_script', methods=['POST'])(upload_interview_script)


@api_bp.route('/available_job_ids', methods=['GET'])
def get_available_job_ids():
    job_ids = vector_store.get_all_job_ids()
    logging.info(f"Available job IDs: {job_ids}")   
    return jsonify({"job_ids": job_ids})


@api_bp.route('/job_matches', methods=['GET'])
def job_matches_route():
    result = get_job_matches()
    logging.info(f"Job matches response: {result}")
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)

@api_bp.route('/job_details/<int:job_id>', methods=['GET'])
def get_job_details(job_id):
    try:
        logging.info(f"Attempting to retrieve job details for ID: {job_id}")
        
        # Get all available job IDs
        all_job_ids = vector_store.get_all_job_ids()
        logging.info(f"All available job IDs: {all_job_ids}")
        
        if job_id not in all_job_ids:
            logging.warning(f"Job ID {job_id} not found in available IDs")
            return jsonify({"error": f"Job with ID {job_id} not found"}), 404

        # Retrieve the job description from your vector store
        job_description = vector_store.get_job_description(job_id)
        if not job_description:
            logging.error(f"Job not found for ID: {job_id}")
            return jsonify({"error": "Job not found"}), 404

        logging.info(f"Successfully retrieved job description for ID: {job_id}")

        try:
            most_recent_resume = vector_store.get_most_recent_resume()
            if not most_recent_resume:
                logging.error("No resume found in the system")
                return jsonify({"error": "No resume found in the system. Please upload a resume."}), 404
        except Exception as e:
            logging.error(f"Error retrieving most recent resume: {str(e)}")
            return jsonify({"error": "Error retrieving resume data. Please try re-uploading your resume."}), 500

        # Get the job details and analysis
        try:
            details_and_analysis = mistral_service.get_job_details_and_analysis(
                job_id, 
                job_description.text_content, 
                most_recent_resume.text_content
            )
        except Exception as e:
            logging.error(f"Error in Mistral service: {str(e)}")
            return jsonify({"error": "Error generating job analysis. Please try again later."}), 500

        if details_and_analysis:
            logging.info(f"Successfully generated job details and analysis for ID: {job_id}")
            return jsonify(details_and_analysis), 200
        else:
            logging.error(f"Failed to retrieve job details for ID: {job_id}")
            return jsonify({"error": "Failed to retrieve job details"}), 500

    except Exception as e:
        logging.error(f"Unexpected error in get_job_details: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500