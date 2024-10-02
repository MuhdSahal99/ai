from flask import Blueprint,jsonify, request, send_file
from .upload import upload_resume, upload_job_description, upload_interview_script
from services.vector_store import vector_store
from models.resume import Resume
from services.job_matching_service import get_job_matches
from services.mistral_service import mistral_service
import logging
import os

logger = logging.getLogger(__name__)

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
    try:
        matches = get_job_matches()
        if isinstance(matches, tuple) and len(matches) == 2:
            return jsonify(matches[0]), matches[1]
        return jsonify(matches)
    except Exception as e:
        logger.error(f"Error in job_matches_route: {str(e)}")
        return jsonify({"error": "An error occurred while fetching job matches"}), 500


@api_bp.route('/job_post_details', methods=['GET'])
def get_job_post_details():
    # Get the most recent job description
    job_description = vector_store.get_most_recent_job_description()
    
    if not job_description:
        return jsonify({"error": "No job descriptions found"}), 404
    
    # Extract job details using Mistral
    job_details = mistral_service.extract_job_details_cached(job_description.text_content)
    
    # Get top 3 matching candidates
    matches = vector_store.search(job_description.text_content, doc_type="resume", k=3)
    
    candidates = []
    for vector_id, similarity_score in matches:
        resume = vector_store.get_resume_by_vector_id(vector_id)
        if resume:
            candidate_details = mistral_service.extract_candidate_details(resume.text_content)
            llm_response = mistral_service.analyze_candidate_fit(job_description.text_content, resume.text_content)
            
            candidates.append({
                "id": str(resume.id),
                "name": candidate_details.get("name", "Unknown"),
                "title": candidate_details.get("title", "Unknown"),
                "llm_response": llm_response,
                "matchingScore": round(similarity_score * 100, 2)
            })
    
    return jsonify({
        "jobTitle": job_details.get("job_title", "Unknown"),
        "companyName": job_details.get("company_name", "Unknown"),
        "candidates": candidates
    })
    

@api_bp.route('/download_cv/<string:resume_id>', methods=['GET'])
def download_cv(resume_id):
    resume = vector_store.get_resume_by_id(resume_id)
    if resume and resume.file_path:
        return send_file(resume.file_path, as_attachment=True, download_name=f"{resume_id}_cv.pdf")
    else:
        return jsonify({"error": "CV not found"}), 404

@api_bp.route('/latest-interview-analysis', methods=['GET'])
def get_latest_interview_analysis():
    logger.info("Received request for latest interview analysis")
    
    interview_script = vector_store.get_most_recent_interview_script()
    # Retrieve the most recent interview script from the vector store
    interview_script = vector_store.get_most_recent_interview_script()
    
    if not interview_script:
        logger.warning("No valid interview scripts found")
        return jsonify({"error": "No valid interview scripts found"}), 404

    logger.info(f"Successfully retrieved latest interview script with ID: {interview_script.id}")
    
    candidate_details = mistral_service.extract_candidate_details(interview_script.text_content)
    # Generate analysis using Mistral service
    analysis = mistral_service.generate_interview_analysis(interview_script.text_content)

    logger.info(f"Analysis generated for latest interview script")

    return jsonify({
        "interview_id": str(interview_script.id),
        "candidate_name": candidate_details.get("name", "Unknown"),
        "candidate_title": candidate_details.get("title", "Unknown"),
        "analysis": analysis
    })