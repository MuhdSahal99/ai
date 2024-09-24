from services.vector_store import vector_store
from services.mistral_service import extract_job_details_batch
import os

def get_job_matches():
    if not vector_store.data.get('resume'):
        return {"error": "No resumes uploaded yet"}, 400
    
    # Get the last uploaded resume
    last_resume = vector_store.data['resume'][-1]
    
    # Perform similarity search
    matches = vector_store.search(last_resume.text_content, doc_type="job_description", k=4)
    
    # Fetch job details and format response
    job_matches = []
    job_descriptions = []
    for vector_id, similarity_score in matches:
        job = next((job for job in vector_store.data['job_description'] if job.vector_id == vector_id), None)
        if job:
            job_descriptions.append(job.text_content)
            percentage_score = similarity_score * 100  # Convert to percentage
            job_matches.append({
                "id": str(job.vector_id),
                "similarityScore": f"{percentage_score:.2f}%",
                "text_content": job.text_content
            })
    
    # Extract job details in batch
    job_details_list = extract_job_details_batch(job_descriptions)
    
    # Merge job details with job matches
    for i, details in enumerate(job_details_list):
        job_matches[i].update({
            "title": details.get('job_title', job_matches[i]['text_content'][:5] + "..."),
            "company": details.get('company_name', 'Unknown'),
            "location": details.get('location', 'Not specified'),
            "salary": details.get('salary', 'Not specified')
        })
    
    # Sort job matches by similarity score in descending order
    job_matches.sort(key=lambda x: float(x['similarityScore'][:-1]), reverse=True)
    
    return job_matches