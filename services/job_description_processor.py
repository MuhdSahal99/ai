import os
import PyPDF2
from dataclasses import dataclass, field
from typing import Optional
import time 

@dataclass
class JobDescription:
    file_path: str
    text_content: str
    id: int = field(default_factory=lambda: int(time.time() * 1000))
    vector_id: Optional[int] = None
    

    def to_dict(self):
        return {
            "id": self.id,
            "file_path": self.file_path,
            "text_content": self.text_content,
            "vector_id": self.vector_id
        }

def process_job_description(file_path: str) -> JobDescription:
    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        # Process the file as a PDF
        text_content = process_pdf(file_path)
    else:
        # Process the file as a plain text file
        text_content = process_text_file(file_path)
    
    return JobDescription(file_path=file_path, text_content=text_content)

def process_pdf(file_path: str) -> str:
    # Use PyPDF2 to extract text from PDF
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text

def process_text_file(file_path: str) -> str:
    try:
        # Read text from a plain text file using UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 encoding if UTF-8 fails
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            return file.read()
