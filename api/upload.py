from flask import request, jsonify
from werkzeug.utils import secure_filename
from services.resume_processor import process_resume
from services.job_description_processor import process_job_description
from services.interview_script_processor import process_interview_script
from services.vector_store import vector_store
from utils.file_handler import process_file
import logging

logger = logging.getLogger(__name__)

def upload_resume():
    return upload_file('resume', process_resume)

def upload_job_description():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            original_filename = secure_filename(file.filename)
            text_content = process_file(file)
            
            # Process the job description
            job_description = process_job_description(text_content, original_filename)
            
            # Add to vector store
            vector_id = vector_store.add_to_vector_store(job_description)
            
            # Update the job description with the vector_id
            job_description.vector_id = vector_id
            
            logger.info(f"Uploaded job description with ID: {job_description.id} and vector_id: {vector_id}")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'job_description': job_description.to_dict()
            }), 200
        except Exception as e:
            logger.error(f"Error in upload_job_description: {str(e)}")
            return jsonify({'error': str(e)}), 500

def upload_interview_script():
    logger.info("Received request to upload interview script")
    logger.debug(f"Request files: {request.files}")
    logger.debug(f"Request form: {request.form}")

    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            original_filename = secure_filename(file.filename)
            text_content = process_file(file)
            
            logger.info(f"File processed successfully")
            
            # Process the interview script
            interview_script = process_interview_script(text_content, original_filename)
            
            logger.info(f"Interview script processed: {interview_script}")
            
            # Add to vector store
            vector_id = vector_store.add_to_vector_store(interview_script)
            
            logger.info(f"Added to vector store with vector_id: {vector_id}")
            
            # Update the interview script with the vector_id
            interview_script.vector_id = vector_id
            
            logger.info(f"Uploaded interview script with ID: {interview_script.id} and vector_id: {vector_id}")
            
            return jsonify({
                'message': 'Interview script uploaded successfully',
                'interview_script': interview_script.to_dict()
            }), 200
        except Exception as e:
            logger.exception(f"Error in upload_interview_script: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    logger.error("Invalid file")
    return jsonify({'error': 'Invalid file'}), 400

def upload_file(file_type, process_function):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            original_filename = secure_filename(file.filename)
            text_content = process_file(file)
            
            # Process the file
            processed_data = process_function(text_content, original_filename)
            
            # Add to vector store
            vector_id = vector_store.add_to_vector_store(processed_data)
            
            # Update the processed data with the vector_id
            processed_data.vector_id = vector_id
            
            return jsonify({
                'message': f'{file_type.capitalize()} uploaded successfully',
                'data': processed_data.to_dict()
            }), 200
        except UnicodeDecodeError as e:
            return jsonify({'error': f'File encoding error: {str(e)}. Please ensure the file is in a standard text format.'}), 400
        except Exception as e:
            logger.error(f"Error in upload_{file_type}: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400