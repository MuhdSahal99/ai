import os
import time
import random
from mistralai import Mistral
from functools import lru_cache
from dotenv import load_dotenv
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class MistralService:
    def __init__(self):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables.")
        self.client = Mistral(api_key=self.api_key)
        self.chat_model = "mistral-large-latest"
        self.max_retries = 5
        self.base_delay = 1  # Base delay in seconds

    @lru_cache(maxsize=100)
    def extract_job_details_cached(self, job_description):
        prompt = f"""
        Extract the following details from the job description:
        - Job Title
        - Company Name
        - Location
        - Salary (if available)

        Job Description:
        {job_description}

        Please return the information in the following format:

        Job Title: [extracted title]
        Company Name: [extracted company name]
        Location: [extracted location]
        Salary: [extracted salary or "Not specified" if not found]
        """

        for attempt in range(self.max_retries):
            try:
                logger.info("Calling Mistral API for job description: %s...", job_description[:60])
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                    }]
                )
                logger.debug("Mistral API Response: %s", response)

                content = response.choices[0].message.content.strip()
                logger.info("Extracted Content: %s", content)

                details = {
                    'job_title': 'Not specified',
                    'company_name': 'Not specified',
                    'location': 'Not specified',
                    'salary': 'Not specified'
                }

                # Parse the content into a dictionary
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        if key in details:
                            details[key] = value.strip()

                logger.info("Parsed Details: %s", details)
                return details

            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message:
                    if attempt == self.max_retries - 1:
                        logger.error("Max retries reached. Error: %s", str(e))
                        return {}

                    delay = (2 ** attempt + random.random()) * self.base_delay
                    logger.warning("Rate limit exceeded. Retrying in %.2f seconds...", delay)
                    time.sleep(delay)
                else:
                    logger.error("Unexpected error calling Mistral API: %s", str(e))
                    return {}

    def extract_job_details_batch(self, job_descriptions):
        return [self.extract_job_details_cached(desc) for desc in job_descriptions]

    def generate_fit_analysis(self, job_description, candidate_resume):
        prompt = f"""
        Based on the following job description and candidate resume, explain how well the candidate fits the role:

        Job Description:
        {job_description}

        Candidate Resume:
        {candidate_resume}

        Provide a detailed analysis in 400 words only. Format your response as follows:

        Overall Fit:
        [Provide a brief overall assessment]

        Strengths:
        - [List key strengths]
        - [Another strength]
        ...

        Areas for Improvement:
        - [List areas where the candidate could improve]
        - [Another area for improvement]
        ...

        Conclusion:
        [Provide a concluding statement]

        Do not use any HTML tags or special formatting in your response.
        """

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                #post-process the content to remove any remaining html-like tags
                processed_content = self.post_process_content(content)
                
                return processed_content

            except Exception as e:
                return "Error: Could not generate analysis."
    def post_process_content(self, content):
        # Remove any HTML-like tags
        
        content = re.sub(r'<[^>]+>', '', content)
        
        # Replace multiple newlines with a single newline
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content

    def get_job_matches_with_llm_response(self, job_descriptions, candidate_resumes):
        matches_with_response = []
        for job_desc, resume in zip(job_descriptions, candidate_resumes):
            job_details = self.extract_job_details_cached(job_desc)
            llm_response = self.generate_fit_analysis(job_desc, resume)

            matches_with_response.append({
                "title": job_details.get("job_title"),
                "company": job_details.get("company_name"),
                "location": job_details.get("location"),
                "salary": job_details.get("salary"),
                "llm_response": llm_response
            })
        return matches_with_response
    
    # employer side 
    def extract_candidate_details(self, resume_text):
        prompt = f"""
        Extract the following details from the resume:
        - Name
        - Title (current or most recent job title)

        Resume:
        {resume_text}

        Please return the information in the following format:

        Name: [extracted name]
        Title: [extracted title]
        """

        response = self.client.chat.complete(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()

        details = {
            'name': 'Unknown',
            'title': 'Unknown'
        }

        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                if key in details:
                    details[key] = value.strip()

        return details
    
    def analyze_candidate_fit(self, job_description, resume_text):
        prompt = f"""
        Analyze how well the candidate fits the job based on the following job description and resume:

        Job Description:
        {job_description}

        Candidate Resume:
        {resume_text}

        Provide a brief analysis (maximum 100 words) of the candidate's fit for the job. 
        Focus on the key strengths and potential areas for improvement.
        Do not use any headings or bullet points in your response.
        """

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                
                # Ensure the response is not too long
                words = content.split()
                if len(words) > 100:
                    content = ' '.join(words[:100]) + '...'
                
                return content

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to generate candidate fit analysis after {self.max_retries} attempts: {str(e)}")
                    return "Unable to generate analysis due to an error."

                delay = (2 ** attempt + random.random()) * self.base_delay
                logger.warning(f"Error in analyze_candidate_fit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                
                
    def extract_candidate_details_from_interview(self, interview_script: str):
        # Define the prompt to extract candidate name and job position
        prompt = f"""
        Extract the following details from the interview script:
        - Candidate's Name
        - Candidate's Position (current or intended job position)

        Interview Script:
        {interview_script}

        Please return the information in the following format:

        Name: [extracted name]
        Position: [extracted position]
        """

        # Try making the API call with retries in case of rate limits or errors
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()

                # Default response if details cannot be extracted
                details = {
                    'name': 'Unknown',
                    'position': 'Unknown'
                }

                # Parse the response to extract name and position
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        if key in details:
                            details[key] = value.strip()

                return details

            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message:
                    if attempt == self.max_retries - 1:
                        logger.error("Max retries reached. Error: %s", str(e))
                        return {}

                    delay = (2 ** attempt + random.random()) * self.base_delay
                    logger.warning("Rate limit exceeded. Retrying in %.2f seconds...", delay)
                    time.sleep(delay)
                else:
                    logger.error("Unexpected error calling Mistral API: %s", str(e))
                    return {}
                
    # for intervew analysis
    def generate_interview_analysis(self, interview_script: str):
        prompts = [
            "Analyze the following interview script and provide insights on the candidate's confidence and personal skills: in 50 words",
            "Based on the interview script, what are the candidate's key strengths? in 50 words",
            "Identify areas where the candidate needs improvement based on the interview script: in 50 words"
        ]

        responses = []
        for prompt in prompts:
            full_prompt = f"{prompt}\n\nInterview Script:\n{interview_script}"

            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.complete(
                        model=self.chat_model,
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    responses.append(response.choices[0].message.content.strip())
                    break  # Exit retry loop if successful

                except Exception as e:
                    error_message = str(e).lower()
                    if "rate limit" in error_message:
                        if attempt == self.max_retries - 1:
                            logging.error("Max retries reached for prompt: %s. Error: %s", prompt, str(e))
                            responses.append("Failed to generate response due to rate limit.")
                        else:
                            delay = (2 ** attempt + random.random()) * self.base_delay
                            logging.warning("Rate limit exceeded for prompt: %s. Retrying in %.2f seconds...", prompt, delay)
                            time.sleep(delay)  # Exponential backoff
                    else:
                        logging.error("Unexpected error calling Mistral API for prompt: %s. Error: %s", prompt, str(e))
                        responses.append("An unexpected error occurred.")

        return {
            "confidence_and_skills": responses[0] if len(responses) > 0 else "No response available.",
            "strengths": responses[1] if len(responses) > 1 else "No response available.",
            "areas_for_improvement": responses[2] if len(responses) > 2 else "No response available."
        }

# Initialize Mistral Service
mistral_service = MistralService()
extract_job_details_batch = mistral_service.extract_job_details_batch
