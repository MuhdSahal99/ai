import PyPDF2
import docx
from models.resume import Resume

def process_resume(file_path: str) -> Resume:
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        text_content = process_pdf(file_path)
    elif file_extension in ['docx', 'doc']:
        text_content = process_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF, DOCX are supported.")
    
    return Resume(id=None, file_path=file_path, text_content=text_content, vector_id=None)

def process_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""  # Add fallback in case of empty text
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")
    return text

def process_docx(file_path: str) -> str:
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        raise ValueError(f"Error processing DOCX: {str(e)}")
    return text