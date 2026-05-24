"""
Task 1.2: Build Assessment Index
Loads catalog.json, normalizes fields, builds inverted index, outputs assessments.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class CatalogIndexer:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.catalog_path = catalog_path
        self.assessments: List = []
        self.keyword_index : defaultdict= defaultdict(set)  # keyword -> set of assessment ids
        self.category_index: defaultdict = defaultdict(set)  # category -> set of assessment ids

    def load_catalog(self) -> List[Dict]:
        """Load raw catalog from JSON"""
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            raw_catalog = json.load(f)
        print(f"✓ Loaded {len(raw_catalog)} raw assessments from catalog")
        return raw_catalog

    def normalize_assessment(self, raw_assessment: Dict) -> Dict:
        """Normalize a single assessment record"""
        return {
            "id": str(raw_assessment.get("id", "")),
            "name": self._clean_name(raw_assessment.get("name", "")),
            "url": raw_assessment.get("url", ""),
            "description": raw_assessment.get("description", ""),
            "job_levels": raw_assessment.get("job_levels", []),
            "keys": raw_assessment.get("keys", []),  # Assessment categories
            "languages": raw_assessment.get("languages", []),
            "duration": raw_assessment.get("duration", ""),
        }

    def _clean_name(self, name: str) -> str:
        """Remove (New) suffix and clean up name"""
        return name.replace(" (New)", "").strip()

    def build_keyword_index(self, assessments: List[Dict]):
        """Build inverted keyword index from assessment names + descriptions"""
        for assessment in assessments:
            assessment_id = assessment["id"]
            
            # Extract keywords from name
            name_tokens = self._tokenize(assessment["name"])
            for token in name_tokens:
                self.keyword_index[token.lower()].add(assessment_id)
            
            # Extract keywords from description
            desc_tokens = self._tokenize(assessment["description"])
            for token in desc_tokens:
                self.keyword_index[token.lower()].add(assessment_id)
            
            # Extract known skills/technologies
            skills = self._extract_skills(assessment["description"])
            for skill in skills:
                self.keyword_index[skill.lower()].add(assessment_id)

    def build_category_index(self, assessments: List[Dict]):
        """Map assessment categories (keys) to assessment ids"""
        for assessment in assessments:
            assessment_id = assessment["id"]
            for key in assessment.get("keys", []):
                self.category_index[key].add(assessment_id)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: split on whitespace and punctuation"""
        if not text:
            return []
        # Split on whitespace, keep alphanumeric + underscores
        tokens = re.findall(r"\b[\w]+\b", text.lower())
        # Filter out common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "is", "are", "was", "were",
            "be", "to", "of", "in", "for", "on", "with", "by", "this", "that"
        }
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    def _extract_skills(self, description: str) -> Set[str]:
        """Extract known skill/technology mentions from description"""
        if not description:
            return set()
        
        # Common skills/technologies to look for
        skills_patterns = {
            "python": r"\bpython\b",
            "java": r"\bjava\b",
            "javascript": r"\bjavascript\b|\bjs\b",
            "sql": r"\bsql\b",
            "rust": r"\brust\b",
            "golang": r"\bgo\b|\bgolang\b",
            "c#": r"\bc#\b|\bcsharp\b",
            "leadership": r"\bleadership\b",
            "communication": r"\bcommunication\b",
            "teamwork": r"\bteamwork\b",
            "aws": r"\baws\b|amazon web services",
            "azure": r"\bazure\b",
            "docker": r"\bdocker\b",
            "kubernetes": r"\bkubernetes\b|k8s",
            "react": r"\breact\b",
            "angular": r"\bangular\b",
            "excel": r"\bexcel\b",
            "word": r"\bword\b",
            "spring": r"\bspring\b",
            "microservices": r"\bmicroservices\b",
            "linux": r"\blinux\b",
            "numeric": r"\bnumeric\b",
            "reasoning": r"\breasoning\b",
            "cognitive": r"\bcognitive\b",
            "personality": r"\bpersonality\b",
            "behavioral": r"\bbehavioral\b",
        }
        
        desc_lower = description.lower()
        found_skills = set()
        for skill, pattern in skills_patterns.items():
            if re.search(pattern, desc_lower):
                found_skills.add(skill)
        return found_skills

    def process_catalog(self) -> List[Dict]:
        """Main processing pipeline"""
        # Load raw catalog
        raw_catalog = self.load_catalog()
        
        # Normalize all assessments
        self.assessments = [self.normalize_assessment(a) for a in raw_catalog]
        print(f"Normalized {len(self.assessments)} assessments")
        
        # Build indices
        self.build_keyword_index(self.assessments)
        print(f"Built keyword index with {len(self.keyword_index)} unique keywords")
        
        self.build_category_index(self.assessments)
        print(f" Built category index with {len(self.category_index)} unique categories")
        
        return self.assessments

    def save_assessments(self, output_path: str = "data/assessments.json"):
        """Save normalized assessments to JSON"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        print(f" Saved {len(self.assessments)} assessments to {output_path}")

    def save_indices(self, output_dir: str = "data"):
        """Save keyword and category indices for retrieval"""
        output_path_keywords = Path(output_dir) / "keyword_index.json"
        output_path_categories = Path(output_dir) / "category_index.json"
        
        # Convert sets to lists for JSON serialization
        keyword_index_serializable = {
            k: list(v) for k, v in self.keyword_index.items()
        }
        category_index_serializable = {
            k: list(v) for k, v in self.category_index.items()
        }
        
        with open(output_path_keywords, "w", encoding="utf-8") as f:
            json.dump(keyword_index_serializable, f, indent=2)
        print(f"Saved keyword index to {output_path_keywords}")
        
        with open(output_path_categories, "w", encoding="utf-8") as f:
            json.dump(category_index_serializable, f, indent=2)
        print(f" Saved category index to {output_path_categories}")


def main():
    print("=" * 60)
    print("Task 1.2: Build Assessment Index")
    print("=" * 60)
    
    indexer = CatalogIndexer()
    assessments = indexer.process_catalog()
    indexer.save_assessments()
    indexer.save_indices()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total assessments: {len(assessments)}")
    print(f"  Unique keywords: {len(indexer.keyword_index)}")
    print(f"  Unique categories: {len(indexer.category_index)}")
    print(f"  Categories: {sorted(indexer.category_index.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
