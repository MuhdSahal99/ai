import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import Union, List, Tuple
from models.resume import Resume
from services.job_description_processor import JobDescription
from services.interview_script_processor import InterviewScript
import logging
import time

class VectorStore:
    def __init__(self, base_path='vectordb'):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.vector_size = 384  # Depends on the model you're using
        self.base_path = base_path
        self.indices = {}
        self.data = {}
        self.load()

    def load(self):
        for doc_type in ['resume', 'job_description', 'interview_script']:
            index_path = os.path.join(self.base_path, f'{doc_type}_index.faiss')
            data_path = os.path.join(self.base_path, f'{doc_type}_data.pkl')
            if os.path.exists(index_path) and os.path.exists(data_path):
                self.indices[doc_type] = faiss.read_index(index_path)
                with open(data_path, 'rb') as f:
                    self.data[doc_type] = pickle.load(f)
            else:
                self.indices[doc_type] = faiss.IndexFlatL2(self.vector_size)
                self.data[doc_type] = []
        self.save()  # Save the empty indices and data lists if they didn't exist

    def save(self):
        os.makedirs(self.base_path, exist_ok=True)
        for doc_type in ['resume', 'job_description', 'interview_script']:
            index_path = os.path.join(self.base_path, f'{doc_type}_index.faiss')
            data_path = os.path.join(self.base_path, f'{doc_type}_data.pkl')
            faiss.write_index(self.indices[doc_type], index_path)
            with open(data_path, 'wb') as f:
                pickle.dump(self.data[doc_type], f)
    
    

    def add_to_vector_store(self, item: Union[Resume, JobDescription, InterviewScript]) -> int:
        doc_type = self._get_doc_type(item)
        vector = self.model.encode([item.text_content])[0]
        vector = vector / np.linalg.norm(vector)
        vector = np.array([vector]).astype('float32')
        self.indices[doc_type].add(vector)
        vector_id = self.indices[doc_type].ntotal - 1
        item.vector_id = vector_id
       
        if doc_type == 'job_description':
            if item.id is None:
                item.id = int(time.time() * 1000)
            logging.info(f"Adding job description with ID: {item.id}")
        elif doc_type == 'resume':
            if not hasattr(item, 'created_at'):
                item.created_at = time.time()
            logging.info(f"Adding resume with created_at: {item.created_at}")

        self.data[doc_type].append(item)
        self.save()
        return vector_id
    
    def get_all_job_ids(self) -> List[int]:
        job_ids = [job.id for job in self.data.get('job_description', []) if job.id is not None]
        logging.info(f"Retrieved job IDs: {job_ids}")
        return job_ids
    
    def get_most_recent_resume(self) -> Union[Resume, None]:
        resumes = self.data.get('resume', [])
        if not resumes:
            return None
        
        current_time = time.time()
        for resume in resumes:
            if not hasattr(resume, 'created_at'):
                resume.created_at = current_time
                logging.warning(f"Resume {resume.id} did not have 'created_at' attribute. Setting to current time.")
        
        most_recent = max(resumes, key=lambda x: x.created_at)
        logging.info(f"Retrieved most recent resume with created_at: {most_recent.created_at}")
        return most_recent
        

    def search(self, query: str, doc_type: str, k: int = 5) -> List[Tuple[int, float]]:
        query_vector = self.model.encode([query])[0]
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize the query vector
        query_vector = np.array([query_vector]).astype('float32')

        # Ensure we're using the correct index for the document type
        index = self.indices[doc_type]
        
        # Perform the search
        distances, indices = index.search(query_vector, k)
        
        # Calculate cosine similarity for each returned index
        results = []
        for i, idx in enumerate(indices[0]):
            vector = index.reconstruct(int(idx))
            cosine_similarity = np.dot(vector, query_vector[0]) / (np.linalg.norm(vector) * np.linalg.norm(query_vector[0]))
            # Ensure similarity scores are between 0 and 1
            cosine_similarity = (cosine_similarity + 1) / 2
            results.append((int(idx), cosine_similarity))
        
        return results
    
    def get_all_items(self, doc_type: str):
        return [item.to_dict() for item in self.data[doc_type]]
    
    

    def get_job_description(self, job_id: int) -> Union[JobDescription, None]:
        job_descriptions = self.data.get('job_description', [])
        logging.info(f"Searching for job ID: {job_id} among {len(job_descriptions)} job descriptions")
        for job in job_descriptions:
            if job.id == job_id:
                logging.info(f"Found job description for ID: {job_id}")
                return job
        logging.warning(f"Job description not found for ID: {job_id}")
        return None
    

    def _get_doc_type(self, item: Union[Resume, JobDescription, InterviewScript]) -> str:
        if isinstance(item, Resume):
            return 'resume'
        elif isinstance(item, JobDescription):
            return 'job_description'
        elif isinstance(item, InterviewScript):
            return 'interview_script'
        else:
            raise ValueError("Unknown item type")
        

vector_store = VectorStore()