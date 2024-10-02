
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import Union, List, Tuple, Optional
from models.resume import Resume
from services.job_description_processor import JobDescription
from services.interview_script_processor import InterviewScript
import logging
import time
import dotenv
from pinecone import Pinecone, ServerlessSpec

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_name='recruiter', dimension=384):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.vector_size = dimension
        self.index_name = index_name
        self.data = {'resume': [], 'job_description': [], 'interview_script': []}
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.vector_size,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                    region=os.getenv('PINECONE_REGION', 'us-west-2')
                )
            )
        
        self.index = self.pc.Index(self.index_name)

    
    
    

    def add_to_vector_store(self, item: Union[Resume, JobDescription, InterviewScript]) -> str:
        try:
            doc_type = self._get_doc_type(item)
            logger.info(f"Adding item of type {doc_type} to vector store")
            
            # Ensure text_content is a string
            text_content = str(item.text_content)
            vector = self.model.encode([text_content])[0].tolist()
            
            # Generate a unique ID for the item
            item_id = f"{doc_type}_{int(time.time() * 1000)}"
            
            # Add metadata
            metadata = {
                "type": doc_type,
                "id": str(item.id),
                "created_at": item.created_at,
                "original_filename": item.original_filename,
                "text_content": text_content  # Store text_content in metadata
            }
            
            # Upsert to Pinecone
            self.index.upsert(vectors=[(item_id, vector, metadata)])
            
            # Update local data structure
            item.vector_id = item_id
            self.data[doc_type].append(item)
            
            logger.info(f"{doc_type.capitalize()} added successfully with vector_id: {item_id}")
            return item_id
        except Exception as e:
            logger.error(f"Error in add_to_vector_store: {str(e)}", exc_info=True)
            raise
            
    
    def get_all_job_ids(self) -> List[int]:
        # Query Pinecone for all job descriptions
        results = self.index.query(
            vector=[0] * self.vector_size,  # Dummy vector
            top_k=100,  # Adjust based on expected maximum number of jobs
            include_metadata=True,
            filter={"type": "job_description"}
        )
        return [int(match.metadata['id']) for match in results.matches if match.metadata['id']]
    
    def get_most_recent_resume(self) -> Union[Resume, None]:
        # Query all resumes and sort by created_at timestamp
        results = self.index.query(
            vector=[0] * self.vector_size,  # Still a dummy vector if needed
            top_k=100,  # Increase the top_k to ensure more results are checked
            include_metadata=True,
            filter={"type": "resume"}
        )
        
        if results.matches:
            # Sort matches by created_at if it's included in metadata
            sorted_matches = sorted(results.matches, key=lambda x: x.metadata.get('created_at', 0), reverse=True)
            metadata = sorted_matches[0].metadata
            return Resume(
                text_content=str(metadata.get('text_content', '')),
                original_filename=metadata.get('original_filename', ''),
                id=int(metadata.get('id', 0)),
                vector_id=sorted_matches[0].id,
                created_at=metadata.get('created_at', 0)
            )
        
        return None
        

    def search(self, query: str, doc_type: str, k: int = 5) -> List[Tuple[str, float]]:
        query_vector = self.model.encode([query])[0].tolist()
        
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter={"type": doc_type}
        )
        
        return [(match.id, match.score) for match in results.matches]
    
    def get_all_items(self, doc_type: str):
        return [item.to_dict() for item in self.data[doc_type]]
    
    

    def get_job_description(self, vector_id: str) -> Union[JobDescription, None]:
        logger.debug(f"Attempting to retrieve job description for vector_id: {vector_id}")
        results = self.index.fetch(ids=[vector_id])
        if results and vector_id in results['vectors']:
            metadata = results['vectors'][vector_id]['metadata']
            logger.debug(f"Retrieved metadata for vector_id {vector_id}: {metadata}")
            if metadata.get('type') == 'job_description':
                job = JobDescription(
                    text_content=str(metadata.get('text_content', '')),
                    original_filename=metadata.get('original_filename', '')
                )
                job.id = int(metadata.get('id', 0))
                job.created_at = metadata.get('created_at', 0)
                job.vector_id = vector_id
                logger.debug(f"Created JobDescription object: {job}")
                return job
            else:
                logger.warning(f"Metadata for vector_id {vector_id} is not a job description")
        else:
            logger.warning(f"No results found for vector_id: {vector_id}")
        return None
    
    
    #employer : most rescent job description
    def get_most_recent_job_description(self) -> Union[JobDescription, None]:
        results = self.index.query(
            vector=[0] * self.vector_size,  # Dummy vector
            top_k=50,
            include_metadata=True,
            filter={"type": "job_description"}
        )
        if not results.matches:
            return None
        most_recent_match = max(results.matches, key=lambda match: int(match.metadata.get('id', 0)))
        return self._fetch_item_from_local_data('job_description', most_recent_match.id)

    # resume vector id .
    def get_resume_by_vector_id(self, vector_id: str) -> Union[Resume, None]:
        return self._fetch_item_from_local_data('resume', vector_id)
    
    # download button 
    def get_resume_by_id(self, resume_id: int) -> Union[Resume, None]:
        results = self.index.query(
            vector=[0] * self.vector_size,  # Dummy vector
            top_k=1,
            include_metadata=True,
            filter={"type": "resume", "id": str(resume_id)}
        )
        if results.matches:
            return self._fetch_item_from_local_data('resume', results.matches[0].id)
        return None
    
    def get_interview_script(self, interview_id: int) -> Union[InterviewScript, None]:
        results = self.index.query(
            vector=[0] * self.vector_size,  # Dummy vector
            top_k=1,
            include_metadata=True,
            filter={"type": "interview_script", "id": str(interview_id)}
        )
        if results.matches:
            return self._fetch_item_from_local_data('interview_script', results.matches[0].id)
        logger.warning(f"Interview script not found for ID: {interview_id}")
        return None
    
    def get_most_recent_interview_script(self) -> Optional[InterviewScript]:
        results = self.index.query(
            vector=[0] * self.vector_size,  # Dummy vector
            top_k=100,
            include_metadata=True,
            filter={"type": "interview_script"}
        )
        logger.info(f"Found {len(results.matches)} interview scripts in total.")

        if not results.matches:
            logger.warning("No interview scripts found.")
            return None

        try:
            most_recent_match = max(results.matches, key=lambda match: int(match.metadata.get('id', 0)))
            logger.info(f"Most recent interview script has ID: {most_recent_match.metadata['id']}")
            return self._fetch_item_from_local_data('interview_script', most_recent_match.id)
        except Exception as e:
            logger.exception(f"Unexpected error in get_most_recent_interview_script: {str(e)}")
            return None
    
    def _fetch_item_from_local_data(self, doc_type: str, vector_id: str) -> Union[Resume, JobDescription, InterviewScript, None]:
        item = next((item for item in self.data[doc_type] if item.vector_id == vector_id), None)
        if item is None:
            # If not found in local data, fetch from Pinecone
            result = self.index.fetch(ids=[vector_id])
            if result and vector_id in result['vectors']:
                metadata = result['vectors'][vector_id]['metadata']
                if doc_type == 'resume':
                    item = Resume(text_content=str(metadata.get('text_content', '')), original_filename=metadata.get('original_filename', ''))
                elif doc_type == 'job_description':
                    item = JobDescription(text_content=str(metadata.get('text_content', '')), original_filename=metadata.get('original_filename', ''))
                elif doc_type == 'interview_script':
                    item = InterviewScript(text_content=str(metadata.get('text_content', '')), original_filename=metadata.get('original_filename', ''))
                item.id = int(metadata.get('id', 0))
                item.created_at = metadata.get('created_at', 0)
                item.vector_id = vector_id
                self.data[doc_type].append(item)
        return item

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