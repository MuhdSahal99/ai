from flask import request, jsonify
from werkzeug.utils import secure_filename
from services.resume_processor import process_resume
from services.job_description_processor import process_job_description
from services.interview_script_processor import process_interview_script
from services.vector_store import vector_store
from utils.file_handler import save_file
import logging

def upload_resume():
    return upload_file('resume', process_resume)

def upload_job_description():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = save_file(file, filename)
        
        # Process the job description
        job_description = process_job_description(file_path)
        
        # Add to vector store
        vector_id = vector_store.add_to_vector_store(job_description)
        
        # Update the job description with the vector_id
        job_description.vector_id = vector_id
        
        logging.info(f"Uploaded job description with ID: {job_description.id} and vector_id: {vector_id}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'job_description': job_description.to_dict()
        }), 200

def upload_interview_script():
    return upload_file('interview_script', process_interview_script)

def upload_file(file_type, process_function):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = save_file(file, filename)
            
            # Process the file
            processed_data = process_function(file_path)
            
            # Add to vector store
            vector_id = vector_store.add_to_vector_store(processed_data)
            
            return jsonify({
                'message': f'{file_type.capitalize()} uploaded successfully',
                'data': processed_data.to_dict()
            }), 200
        except UnicodeDecodeError as e:
            return jsonify({'error': f'File encoding error: {str(e)}. Please ensure the file is in a standard text format.'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400