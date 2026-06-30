"""
Context Extractor: Parse conversation to extract intent, constraints, and metadata
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ExtractedContext:
    """Extracted context from conversation"""
    intent: str  # What is the hiring need?
    constraints: Dict[str, Any]  # Role, seniority, skills, industries
    turn_count: int  # How many turns so far?
    prior_recommendations: List[str]  # Previous assessment names recommended
    conversation_stage: str  # "clarifying", "recommending", "refining", "comparing"


class ContextExtractor:
    """Extract intent, constraints, and metadata from conversation history"""
    
    # Common seniority keywords
    SENIORITY_KEYWORDS = {
        "entry": ["entry", "graduate", "junior", "entry-level", "fresh"],
        "mid": ["mid", "intermediate", "professional", "experienced", "3-5 years"],
        "senior": ["senior", "lead", "principal", "10+ years", "15+ years"],
        "executive": ["executive", "cxo", "director", "c-level", "vp", "cfo", "ceo"],
    }
    
    # Common job level keywords
    JOB_LEVEL_KEYWORDS = {
        "individual_contributor": ["individual contributor", "ic", "engineer", "analyst", "specialist"],
        "manager": ["manager", "team lead", "supervisor", "front line manager"],
        "director": ["director", "senior manager", "head of"],
        "executive": ["executive", "cxo", "c-level", "vp"],
    }
    
    # Common role keywords
    ROLE_KEYWORDS = {
        "leadership": ["leadership", "leader", "manage", "lead", "executive"],
        "technical": ["engineer", "developer", "programmer", "architect", "technical"],
        "sales": ["sales", "rep", "account", "business development"],
        "customer_service": ["customer service", "support", "contact centre", "call center"],
        "finance": ["finance", "accounting", "analyst", "accountant"],
        "operational": ["operational", "operations", "process"],
    }
    
    def __init__(self):
        pass
    
    def extract(self, messages: List[Dict[str, str]]) -> ExtractedContext:
        """
        Extract context from conversation history
        
        Args:
            messages: List of {role, content} dicts
        
        Returns:
            ExtractedContext with intent, constraints, turn_count, etc.
        """
        
        # Build combined text for analysis
        full_text = " ".join([msg["content"] for msg in messages])
        text_lower = full_text.lower()
        
        # Extract intent
        intent = self._extract_intent(messages, text_lower)
        
        # Extract constraints
        constraints = self._extract_constraints(text_lower)
        
        # Count turns (only user messages count as turns)
        turn_count = sum(1 for m in messages if m["role"] == "user")
        
        # Extract prior recommendations (from assistant messages)
        prior_recommendations = self._extract_prior_recommendations(messages)
        
        # Determine conversation stage
        conversation_stage = self._determine_stage(intent, constraints, turn_count)
        
        return ExtractedContext(
            intent=intent,
            constraints=constraints,
            turn_count=turn_count,
            prior_recommendations=prior_recommendations,
            conversation_stage=conversation_stage,
        )
    
    def _extract_intent(self, messages: List[Dict[str, str]], text_lower: str) -> str:
        """Extract hiring need/intent from conversation"""
        
        # Get latest user message as primary intent
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        if not user_messages:
            return "hiring assessment"
        
        latest_user = user_messages[-1].lower()
        
        # Look for role keywords in latest message
        for role_type, keywords in self.ROLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in latest_user:
                    return f"{role_type} role assessment"
        
        # Fallback: use first 100 chars of latest user message
        return latest_user[:100] if latest_user else "hiring assessment"
    
    def _extract_constraints(self, text_lower: str) -> Dict[str, Any]:
        """Extract role, seniority, skills, industries from text"""
        
        constraints = {}
        
        # Extract seniority
        seniority = None
        for level, keywords in self.SENIORITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                seniority = level
                break
        if seniority:
            constraints["seniority"] = seniority
        
        # Extract job levels
        job_levels = []
        for level, keywords in self.JOB_LEVEL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                job_levels.append(level)
        if job_levels:
            constraints["job_levels"] = list(set(job_levels))
        
        # Extract role types
        roles = []
        for role_type, keywords in self.ROLE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                roles.append(role_type)
        if roles:
            constraints["roles"] = list(set(roles))
        
        # Extract skills (look for programming languages, tools, soft skills)
        skills = self._extract_skills(text_lower)
        if skills:
            constraints["skills"] = skills
        
        # Extract industries (look for common industry keywords)
        industries = self._extract_industries(text_lower)
        if industries:
            constraints["industries"] = industries
        
        # Extract languages (look for language mentions)
        languages = self._extract_languages(text_lower)
        if languages:
            constraints["languages"] = languages
        
        # Extract test types (cognitive, personality, SJT, etc.)
        test_types = self._extract_test_types(text_lower)
        if test_types:
            constraints["test_types"] = test_types
        
        return constraints
    
    def _extract_skills(self, text_lower: str) -> List[str]:
        """Extract technical and soft skills from text"""
        
        skill_keywords = {
            "python": ["python"],
            "java": ["java"],
            "javascript": ["javascript", "typescript", "node"],
            "sql": ["sql", "database", "postgres", "mysql"],
            "leadership": ["leadership", "lead", "manage"],
            "communication": ["communication", "communication skills", "speaking"],
            "teamwork": ["teamwork", "team", "collaboration", "collaborate"],
            "problem_solving": ["problem solving", "problem-solving", "analytical"],
            "rust": ["rust"],
            "golang": ["golang", "go"],
            "aws": ["aws", "amazon web services"],
            "azure": ["azure"],
            "linux": ["linux"],
            "docker": ["docker"],
        }
        
        found_skills = []
        for skill, keywords in skill_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_industries(self, text_lower: str) -> List[str]:
        """Extract industry mentions from text"""
        
        industry_keywords = {
            "finance": ["finance", "banking", "financial", "fintech"],
            "tech": ["tech", "technology", "software", "it"],
            "healthcare": ["healthcare", "health", "medical", "hospital"],
            "retail": ["retail", "sales", "e-commerce"],
            "manufacturing": ["manufacturing", "industrial", "factory"],
            "consulting": ["consulting", "consultant"],
        }
        
        found_industries = []
        for industry, keywords in industry_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_industries.append(industry)
        
        return found_industries
    
    def _extract_languages(self, text_lower: str) -> List[str]:
        """Extract language requirements from text"""
        
        language_keywords = {
            "english": ["english"],
            "spanish": ["spanish"],
            "french": ["french"],
            "german": ["german"],
            "chinese": ["chinese", "mandarin"],
            "bilingual": ["bilingual", "multilingual"],
        }
        
        found_languages = []
        for lang, keywords in language_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_languages.append(lang)
        
        return found_languages
    
    def _extract_test_types(self, text_lower: str) -> List[str]:
        """Extract test type requirements from text (cognitive, personality, SJT, etc.)"""
        
        test_type_keywords = {
            "cognitive": ["cognitive", "cognitive ability", "cognitive test", "numerical", "verbal", "reasoning"],
            "personality": ["personality", "personality test", "personality measure", "behavioral"],
            "situational_judgement": ["situational judgement", "situational judgment", "sjt", "scenario"],
            "leadership": ["leadership", "leadership potential", "management"],
            "skills": ["skills test", "skill assessment", "skills"],
        }
        
        found_types = []
        for test_type, keywords in test_type_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_types.append(test_type)
        
        return found_types
    
    def _extract_prior_recommendations(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract assessment names from prior recommendations"""
        
        recommendations = []
        
        # Look for assessment names in assistant messages
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                
                # Simple heuristic: look for capitalized phrases with assessment-like patterns
                # This would be improved with actual assessment names from catalog
                patterns = [
                    r"(?:the |a |an )?([A-Z][a-zA-Z\s\(\)0-9\-\.&]+?)(?:is |are |test|assessment|questionnaire|report)",
                    r"\|.*?\|.*?\|.*?\n.*?\|.*?\n.*?\|.*?\|.*?\n",  # Table-like structure
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    recommendations.extend([m.strip() for m in matches])
        
        # Remove duplicates and very short strings
        recommendations = [r for r in recommendations if len(r) > 3]
        return list(set(recommendations))[:10]  # Limit to 10
    
    def _determine_stage(
        self,
        intent: str,
        constraints: Dict[str, Any],
        turn_count: int,
    ) -> str:
        """Determine conversation stage based on context"""
        
        if turn_count >= 8:
            return "ending"
        elif constraints and len(constraints) >= 2:
            return "recommending"
        elif constraints and len(constraints) == 1:
            return "clarifying"
        else:
            return "initial"


if __name__ == "__main__":
    extractor = ContextExtractor()
    
    # Test with sample conversation
    test_messages = [
        {"role": "user", "content": "We're hiring a senior Java engineer for our AWS infrastructure team"},
        {"role": "assistant", "content": "Great! To narrow this down, what's the seniority level and focus area?"},
        {"role": "user", "content": "Senior IC, they'll own microservices architecture. We need Java, Spring, and SQL expertise"},
    ]
    
    context = extractor.extract(test_messages)
    print(f"\nExtracted Context:")
    print(f"  Intent: {context.intent}")
    print(f"  Constraints: {context.constraints}")
    print(f"  Turn count: {context.turn_count}")
    print(f"  Stage: {context.conversation_stage}")
