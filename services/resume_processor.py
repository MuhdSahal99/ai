import logging
from models.resume import Resume

logger = logging.getLogger(__name__)

def process_resume(text_content: str, original_filename: str) -> Resume:
    logger.info(f"Processing resume: {original_filename}")
    resume = Resume(text_content=text_content, original_filename=original_filename)
    logger.info(f"Created Resume object: {resume}")
    return resume