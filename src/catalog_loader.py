"""
CatalogLoader: Load and manage the assessment catalog with FAISS indexing
"""

import json
from typing import List, Dict, Optional
from pathlib import Path


class CatalogLoader:
    """Load and query assessments with FAISS-backed retrieval"""
    
    def __init__(self, assessments_path: str = "data/assessments.json"):
        self.assessments_path = assessments_path
        self.assessments : List= []
        self.assessment_by_id: Dict = {}
        self.assessment_by_name: Dict = {}
        self._load()
    
    def _load(self):
        """Load assessments from JSON"""
        with open(self.assessments_path, "r", encoding="utf-8") as f:
            self.assessments = json.load(f)
        
        # Build lookup maps
        for assessment in self.assessments:
            assessment_id = assessment["id"]
            self.assessment_by_id[assessment_id] = assessment
            
            # Case-insensitive name lookup
            name_lower = assessment["name"].lower()
            self.assessment_by_name[name_lower] = assessment
    
    def get_by_id(self, assessment_id: str) -> Optional[Dict]:
        """Get assessment by ID"""
        return self.assessment_by_id.get(assessment_id)
    
    def get_by_name(self, name: str) -> Optional[Dict]:
        """Get assessment by name (case-insensitive)"""
        return self.assessment_by_name.get(name.lower())
    
    def get_all(self) -> List[Dict]:
        """Get all assessments"""
        return self.assessments
    
    def validate_url(self, url: str) -> bool:
        """Check if URL is from catalog"""
        return any(a["url"] == url for a in self.assessments)
    
    def validate_urls(self, urls: List[str]) -> bool:
        """Check if all URLs are from catalog"""
        valid_urls = {a["url"] for a in self.assessments}
        return all(url in valid_urls for url in urls)
    
    def get_assessment_names(self) -> List[str]:
        """Get all assessment names"""
        return [a["name"] for a in self.assessments]
    
    def get_by_category(self, category: str) -> List[Dict]:
        """Get assessments by category (keys)"""
        return [a for a in self.assessments if category in a.get("keys", [])]
    
    def get_by_job_level(self, job_level: str) -> List[Dict]:
        """Get assessments by job level"""
        return [a for a in self.assessments if job_level in a.get("job_levels", [])]
    
    def get_by_language(self, language: str) -> List[Dict]:
        """Get assessments by language"""
        return [a for a in self.assessments if language in a.get("languages", [])]
    
    def get_summary(self) -> Dict:
        """Get catalog summary statistics"""
        categories = set()
        job_levels = set()
        languages = set()
        
        for a in self.assessments:
            categories.update(a.get("keys", []))
            job_levels.update(a.get("job_levels", []))
            languages.update(a.get("languages", []))
        
        return {
            "total_assessments": len(self.assessments),
            "categories": sorted(categories),
            "job_levels": sorted(job_levels),
            "languages": sorted(languages),
            "num_categories": len(categories),
            "num_job_levels": len(job_levels),
            "num_languages": len(languages),
        }


if __name__ == "__main__":
    loader = CatalogLoader()
    
    print("=" * 60)
    print("CatalogLoader Summary")
    print("=" * 60)
    
    summary = loader.get_summary()
    print(f"\nTotal assessments: {summary['total_assessments']}")
    print(f"Categories: {summary['num_categories']}")
    print(f"  {', '.join(summary['categories'][:3])}...")
    print(f"Job levels: {summary['num_job_levels']}")
    print(f"Languages: {summary['num_languages']}")
    
    print(f"\nSample assessments:")
    for a in loader.get_all()[:3]:
        print(f"  - {a['name']}")
