import os
import time
import random
from mistralai import Mistral
from functools import lru_cache
from dotenv import load_dotenv
import logging
import json

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
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
                logger.debug("Mistral API Response: %s", response)

                content = response.choices[0].message.content.strip()
                logger.info("Extracted Content: %s", content)

                # Initialize details with default values
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
    
    def get_job_details_and_analysis(self, job_id, job_description, resume_text):
        prompt = f"""
        Given the following job description and resume, provide a detailed analysis of the job and how well the candidate fits the role. Include the following sections:
        analyse the given reume and job description properly 
        1. Job Details:
           - Title
           - Company Name
           - Location
           - Salary (if available)
           - A summary of key responsibilities and requirements

        2. Fit Analysis:
           - An assessment of how well the candidate's skills and experience match the job requirements
           - Highlights of the candidate's strengths relevant to this role
           - Areas areas 
           - Overall suitability score (as a percentage)

        Job Description:
        {job_description}

        Candidate's Resume:
        {resume_text}

        Please provide your analysis in a structured json format that can be easily parsed into sections.
        """

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Mistral API for job details and analysis for job ID: {job_id}")
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.choices[0].message.content.strip()
                logger.info(f"Generated content for job ID {job_id}: {content[:100]}...")

                if not content:
                    logger.error("No content returned from Mistral API.")
                    return {
                        "job_id": job_id,
                        "analysis": {"fit_analysis": {}, "job_details": {}}
                    }

                # Parse the content into structured data
                parsed_content = self.parse_job_analysis(content)

                return {
                    "job_id": job_id,
                    "analysis": parsed_content
                }

            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message:
                    if attempt == self.max_retries - 1:
                        logger.error("Max retries reached. Error: %s", str(e))
                        return None

                    delay = (2 ** attempt + random.random()) * self.base_delay
                    logger.warning("Rate limit exceeded. Retrying in %.2f seconds...", delay)
                    time.sleep(delay)
                else:
                    logger.error("Unexpected error calling Mistral API: %s", str(e))
                    return None

        return None  # If all retries fail

    def parse_job_analysis(self, content):
    # Log the raw content for debugging
        logger.info(f"Raw content from API: {content[:100]}...")

        # Attempt to clean and sanitize the content
        content = content.strip()

        # Initialize a dictionary to store the parsed content
        parsed = {
            "job_details": {},
            "fit_analysis": {}
        }

        # Attempt to extract JSON from the content
        try:
            # Try to detect and extract JSON from any surrounding text
            start_index = content.find('{')
            end_index = content.rfind('}')
            if start_index != -1 and end_index != -1:
                content_json = json.loads(content[start_index:end_index + 1])
            else:
                logger.error(f"No valid JSON found in the content: {content}")
                return parsed  # Return empty parsed structure on error
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from content: {str(e)}")
            return parsed  # Return empty parsed structure on error

        # Extract Job Details
        if "JobDetails" in content_json:
            job_details = content_json["JobDetails"]
            parsed["job_details"] = {
                "job_title": job_details.get("Title", "Not specified"),
                "company_name": job_details.get("CompanyName", "Not specified"),
                "location": job_details.get("Location", "Not specified"),
                "salary": job_details.get("Salary", "Not specified"),
                "key_responsibilities": job_details.get("Key Responsibilities", "Not specified")
            }

        # Extract Fit Analysis
        if "FitAnalysis" in content_json:
            fit_analysis = content_json["FitAnalysis"]
            parsed["fit_analysis"] = {
                "overall_suitability_score": fit_analysis.get("Overall Suitability", "Not specified"),
                "candidate_strengths": fit_analysis.get("Strengths", "Not specified"),
                "areas_for_improvement": fit_analysis.get("Areas for Improvement", "Not specified"),
                "detailed_analysis": fit_analysis.get("Detailed Analysis", "Not specified")
            }

        logger.info(f"Parsed job analysis: {parsed}")
        return parsed

mistral_service = MistralService()

def extract_job_details_batch(job_descriptions):
    return mistral_service.extract_job_details_batch(job_descriptions)
    


