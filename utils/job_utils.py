import logging
from services.vector_store import VectorStore
import time

def check_job_ids(vector_store: VectorStore):
    job_ids = vector_store.get_all_job_ids()
    issues_found = False
    for i, job_id in enumerate(job_ids):
        if job_id is None:
            logging.warning(f"Job description at index {i} has no ID. This shouldn't happen with the new setup.")
            issues_found = True
    
    if not issues_found:
        logging.info("All job descriptions have valid IDs.")
    else:
        logging.warning("Some job descriptions have missing IDs. Consider regenerating the vector store.")
    
    logging.info(f"All job IDs in vector store: {job_ids}")
    return job_ids

def regenerate_job_ids(vector_store: VectorStore):
    job_ids = check_job_ids(vector_store)
    regenerated_count = 0
    for i, job_id in enumerate(job_ids):
        if job_id is None:
            new_id = str(int(time.time() * 1000) + i)  # Use current timestamp plus index to ensure uniqueness
            vector_store.update_job_id(i, new_id)
            logging.info(f"Generated new ID {new_id} for job description at index {i}")
            regenerated_count += 1
    
    if regenerated_count > 0:
        logging.info(f"Regenerated {regenerated_count} job IDs. Vector store has been updated.")
    else:
        logging.info("No job IDs needed regeneration.")

    # Log all job IDs after regeneration
    job_ids = vector_store.get_all_job_ids()
    logging.info(f"All job IDs in vector store after regeneration: {job_ids}")
    return job_ids

def check_and_fix_job_ids(vector_store: VectorStore):
    initial_job_ids = check_job_ids(vector_store)
    updated_job_ids = regenerate_job_ids(vector_store)

    logging.info(f"Initial job IDs: {initial_job_ids}")
    logging.info(f"Final job IDs after potential regeneration: {updated_job_ids}")

    return updated_job_ids

# New method to be added to VectorStore class
def update_job_id(self, index: int, new_id: str):
    results = self.index.query(
        vector=[0] * self.vector_size,
        top_k=1,
        include_metadata=True,
        filter={"type": "job_description"}
    )
    if results.matches:
        vector_id = results.matches[0].id
        current_metadata = results.matches[0].metadata
        current_metadata['id'] = new_id
        self.index.update(id=vector_id, set_metadata=current_metadata)
        logging.info(f"Updated job ID for vector {vector_id} to {new_id}")
    else:
        logging.warning(f"No job description found at index {index}")