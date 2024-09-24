import logging
from services.vector_store import VectorStore
import time

def check_job_ids(vector_store: VectorStore):
    job_descriptions = vector_store.data.get('job_description', [])
    issues_found = False
    for i, job in enumerate(job_descriptions):
        if job.id is None:
            logging.warning(f"Job description at index {i} has no ID. This shouldn't happen with the new setup.")
            issues_found = True
    
    if not issues_found:
        logging.info("All job descriptions have valid IDs.")
    else:
        logging.warning("Some job descriptions have missing IDs. Consider regenerating the vector store.")
    
    # Log all job IDs
    job_ids = [job.id for job in job_descriptions]
    logging.info(f"All job IDs in vector store: {job_ids}")

def regenerate_job_ids(vector_store: VectorStore):
    job_descriptions = vector_store.data.get('job_description', [])
    regenerated_count = 0
    for i, job in enumerate(job_descriptions):
        if job.id is None:
            job.id = int(time.time() * 1000) + i  # Use current timestamp plus index to ensure uniqueness
            logging.info(f"Generated new ID {job.id} for job description at index {i}")
            regenerated_count += 1
    
    if regenerated_count > 0:
        vector_store.save()
        logging.info(f"Regenerated {regenerated_count} job IDs. Vector store has been updated and saved.")
    else:
        logging.info("No job IDs needed regeneration.")

    # Log all job IDs after regeneration
    job_ids = [job.id for job in job_descriptions]
    logging.info(f"All job IDs in vector store after regeneration: {job_ids}")

def check_and_fix_job_ids(vector_store: VectorStore):
    check_job_ids(vector_store)
    regenerate_job_ids(vector_store)

    # Final check and log
    job_descriptions = vector_store.data.get('job_description', [])
    job_ids = [job.id for job in job_descriptions]
    logging.info(f"Final list of all job IDs in vector store: {job_ids}")

    return job_ids