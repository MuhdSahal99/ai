from services.vector_store import vector_store
from services.mistral_service import mistral_service
import logging

logger = logging.getLogger(__name__)

def get_job_matches():
    try:
        if not vector_store.data.get('resume'):
            logger.warning("No resumes found in vector store")
            return {"error": "No resumes uploaded yet"}, 400
        
        # Get the last uploaded resume
        last_resume = vector_store.get_most_recent_resume()
        if not last_resume:
            logger.warning("Failed to retrieve the most recent resume")
            return {"error": "Failed to retrieve the most recent resume"}, 500

        logger.info(f"Last uploaded resume: {last_resume.original_filename}")
        
        # Perform similarity search
        matches = vector_store.search(last_resume.text_content, doc_type="job_description", k=4)
        logger.info(f"Found {len(matches)} matches")
        
        # Fetch job details and format response
        job_matches = []
        job_descriptions = []
        for vector_id, similarity_score in matches:
            logger.debug(f"Processing match with vector_id: {vector_id}")
            job = vector_store.get_job_description(vector_id)
            if job:
                logger.debug(f"Retrieved job description: {job.id}")
                job_descriptions.append(job.text_content)
                formatted_score = f"{similarity_score * 100:.2f}%"
                job_matches.append({
                    "id": str(job.id),
                    "similarityScore": formatted_score,
                    "text_content": job.text_content
                })
            else:
                logger.warning(f"Failed to retrieve job description for vector_id: {vector_id}")
        
        logger.info(f"Extracted {len(job_descriptions)} job descriptions")
        
        if not job_descriptions:
            logger.warning("No job descriptions extracted. Returning empty result.")
            return []

        # Use Mistral service to get job details and LLM responses
        detailed_matches = mistral_service.get_job_matches_with_llm_response(job_descriptions, [last_resume.text_content] * len(job_descriptions))
        
        # Merge detailed information with job matches
        for i, detailed_match in enumerate(detailed_matches):
            job_matches[i].update({
                "title": detailed_match.get("title", "Unknown Title"),
                "company": detailed_match.get("company", "Unknown Company"),
                "location": detailed_match.get("location", "Unknown Location"),
                "salary": detailed_match.get("salary", "Not specified"),
                "llm_response": detailed_match.get("llm_response", "")
            })
        
        logger.info(f"Returning {len(job_matches)} job matches")
        return job_matches

    except Exception as e:
        logger.error(f"Unexpected error in get_job_matches: {str(e)}", exc_info=True)
        return {"error": "An unexpected error occurred"}, 500