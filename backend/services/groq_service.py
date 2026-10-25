import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from groq import Groq
from groq.types.chat import ChatCompletion

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class CareerRecommendation:
    """Structured career recommendation from Groq."""
    skills_to_learn: List[str]
    learning_roadmap: List[Dict[str, str]]
    portfolio_projects: List[str]
    certifications: List[str]
    career_suggestions: List[str]
    limitations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'skills_to_learn': self.skills_to_learn,
            'learning_roadmap': self.learning_roadmap,
            'portfolio_projects': self.portfolio_projects,
            'certifications': self.certifications,
            'career_suggestions': self.career_suggestions,
            'limitations': self.limitations
        }


class GroqService:
    """Backend-only Groq integration for career recommendations."""
    
    def __init__(self):
        self.client = None
        self.model = "llama-3.1-8b-instant"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client if API key is available."""
        self.api_key = os.getenv('GROQ_API_KEY')
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self.client = None
        else:
            self.client = None
            logger.info("GROQ_API_KEY not set. AI recommendations unavailable.")
    
    def is_available(self) -> bool:
        """Check if Groq service is configured and available."""
        if self.client is None and os.getenv('GROQ_API_KEY'):
            self._initialize_client()
        return self.client is not None
    
    def _build_prompt(
        self,
        profile: Dict[str, Any],
        predicted_salary: float,
        currency: str,
        model_name: str,
        dataset_limitations: str
    ) -> str:
        """Build the prompt for Groq."""
        return f"""You are a career advisor for software engineers. Provide personalized career recommendations based on the user's profile and predicted salary.

USER PROFILE:
- Years of Experience: {profile.get('years_experience', 'N/A')}
- Education Level: {profile.get('education_level', 'N/A')}
- Target Job Title: {profile.get('job_title', 'N/A')}
- Current Skills: {', '.join(profile.get('skills', [])) if profile.get('skills') else 'None specified'}
- Programming Languages: {', '.join(profile.get('programming_languages', [])) if profile.get('programming_languages') else 'None specified'}
- Country: {profile.get('country', 'N/A')}

PREDICTED SALARY: {predicted_salary:,.0f} {currency}
MODEL USED: {model_name}

DATASET LIMITATIONS: {dataset_limitations}

IMPORTANT: The predicted salary is an ESTIMATE from a machine learning model trained on synthetic educational data. It is NOT a guaranteed salary offer. Do not make specific claims about salary increases from learning specific skills.

Return a JSON object with exactly these fields:
{{
  "skills_to_learn": ["skill1", "skill2", ...],
  "learning_roadmap": [
    {{"priority": "High", "topic": "topic name", "description": "brief description"}},
    {{"priority": "Medium", "topic": "topic name", "description": "brief description"}}
  ],
  "portfolio_projects": ["project idea 1", "project idea 2", ...],
  "certifications": ["certification 1", "certification 2", ...],
  "career_suggestions": ["suggestion 1", "suggestion 2", ...],
  "limitations": ["limitation 1", "limitation 2", ...]
}}

Guidelines:
- Skills should be relevant to the target role and current market demands
- Learning roadmap should be prioritized (High/Medium/Low)
- Portfolio projects should be realistic and demonstrable
- Certifications should be recognized in the industry
- Career suggestions should be actionable
- Limitations MUST include: prediction uncertainty, synthetic data, market variability, individual negotiation factors
- Keep each list to 3-6 items
- Be concise and practical"""
    
    def _parse_response(self, response_text: str) -> CareerRecommendation:
        """Parse and validate Groq response."""
        try:
            # Extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            json_str = response_text[start:end]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'skills_to_learn', 'learning_roadmap', 'portfolio_projects',
                'certifications', 'career_suggestions', 'limitations'
            ]
            for field in required_fields:
                if field not in data:
                    data[field] = []
            
            # Ensure lists
            for field in required_fields:
                if not isinstance(data[field], list):
                    data[field] = []
            
            return CareerRecommendation(**data)
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse Groq response: {e}")
            # Return default fallback
            return CareerRecommendation(
                skills_to_learn=[],
                learning_roadmap=[],
                portfolio_projects=[],
                certifications=[],
                career_suggestions=[],
                limitations=["Failed to parse AI response. Please try again."]
            )
    
    def get_recommendations(
        self,
        profile: Dict[str, Any],
        predicted_salary: float,
        currency: str,
        model_name: str
    ) -> Optional[CareerRecommendation]:
        """Get career recommendations from Groq."""
        if not self.is_available():
            return None
        
        dataset_limitations = (
            "The salary prediction model was trained on a small synthetic educational dataset (~100 records) "
            "that does not represent real-world salary distributions. Predictions may not generalize to "
            "specific companies, regions, or market conditions. The model does not account for bonuses, "
            "equity, benefits, company size, industry, remote work policies, or individual negotiation."
        )
        
        prompt = self._build_prompt(
            profile, predicted_salary, currency, model_name, dataset_limitations
        )
        
        CANDIDATE_MODELS = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.1-8b-instant"
        ]
        
        last_error = None
        for candidate_model in CANDIDATE_MODELS:
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {"role": "system", "content": "You are a career advisor. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                response_text = response.choices[0].message.content
                return self._parse_response(response_text)
            except Exception as e:
                last_error = e
                logger.warning(f"Groq model {candidate_model} failed: {e}. Trying next fallback...")
                continue
                
        logger.error(f"All Groq candidate models failed. Last error: {last_error}")
        return CareerRecommendation(
            skills_to_learn=[],
            learning_roadmap=[],
            portfolio_projects=[],
            certifications=[],
            career_suggestions=[],
            limitations=[f"AI service error: {str(last_error)}"]
        )


# Singleton instance
groq_service = GroqService()