import io
import PyPDF2
import docx
from werkzeug.utils import secure_filename

def process_file(file):
    """
    Process the uploaded file in memory and return its content as text.
    
    :param file: File object from request.files
    :return: Extracted text content from the file
    """
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if file_extension == 'pdf':
        return process_pdf(file)
    elif file_extension in ['docx', 'doc']:
        return process_docx(file)
    elif file_extension in ['txt', 'text']:
        return file.read().decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def process_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])