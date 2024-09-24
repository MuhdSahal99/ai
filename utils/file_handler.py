import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')

def save_file(file, filename):
    # Ensure the filename is secure to avoid directory traversal attacks
    filename = secure_filename(filename)
    
    # Create the upload directory if it does not exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Generate the full file path
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # Save the file
        file.save(file_path)
    except Exception as e:
        raise ValueError(f"Error saving file: {str(e)}")
    
    # Return the absolute path for consistency
    return os.path.abspath(file_path)
