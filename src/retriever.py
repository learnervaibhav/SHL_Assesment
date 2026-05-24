"""
Retriever: Hybrid Keyword + Semantic Search using FAISS
Combines TF-IDF/keyword matching with semantic similarity for effective retrieval
"""

import json
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import sys
from functools import lru_cache


class HybridRetriever:
    """
    Combines keyword-based retrieval (from keyword index) with semantic retrieval (from FAISS)
    for a hybrid search that balances precision and recall
    """
    
    def __init__(
        self,
        assessments_path: str = "data/assessments.json",
        keyword_index_path: str = "data/keyword_index.json",
        category_index_path: str = "data/category_index.json",
        faiss_index_path: str = "data/assessments_index.faiss",
        id_mapping_path: str = "data/id_mapping.json",
    ):
        """Initialize retriever with all necessary indices"""
        self.assessments_path = assessments_path
        self.keyword_index_path = keyword_index_path
        self.category_index_path = category_index_path
        self.faiss_index_path = faiss_index_path
        self.id_mapping_path = id_mapping_path
        
        self.assessments: Dict = {}
        self.assessment_list: list = []
        self.keyword_index : Dict= {}
        self.category_index: Dict = {}
        self.faiss_index = None
        self.id_mapping: Dict = {}
        self.reverse_id_mapping : Dict= {}
        self.embedding_model = None
        self.query_cache: Dict[str, List[Dict]] = {}  # Simple cache for frequent queries
        self.max_cache_size = 50  # Limit cache to 50 queries
        
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data and indices"""
        print("Initializing HybridRetriever...")
        
        # Load assessments
        with open(self.assessments_path, "r", encoding="utf-8") as f:
            self.assessment_list = json.load(f)
            self.assessments = {a["id"]: a for a in self.assessment_list}
        print(f"   Loaded {len(self.assessments)} assessments")
        
        # Load keyword index
        with open(self.keyword_index_path, "r", encoding="utf-8") as f:
            keyword_index_data = json.load(f)
            # Convert lists back to sets for faster lookup
            self.keyword_index = {k: set(v) for k, v in keyword_index_data.items()}
        print(f"   Loaded keyword index with {len(self.keyword_index)} keywords")
        
        # Load category index
        with open(self.category_index_path, "r", encoding="utf-8") as f:
            category_index_data = json.load(f)
            self.category_index = {k: set(v) for k, v in category_index_data.items()}
        print(f"  Loaded category index with {len(self.category_index)} categories")
        
        # Load FAISS index
        try:
            import faiss  # type: ignore
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            print(f"   Loaded FAISS index ({self.faiss_index.ntotal} vectors)")
        except ImportError:
            print("   FAISS not available, semantic search disabled")
        except FileNotFoundError:
            print(f"   FAISS index not found at {self.faiss_index_path}")
        
        # Load ID mapping
        with open(self.id_mapping_path, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
            self.reverse_id_mapping = {v: k for k, v in self.id_mapping.items()}
        print(f"   Loaded ID mapping for {len(self.id_mapping)} assessments")
    
    def _init_embedding_model(self):
        """Lazy-load embedding model only when needed"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                print(" sentence-transformers not available")
                return False
        return True
    
    def _keyword_search(self, query: str) -> Dict[str, float]:
        """
        Keyword-based search using inverted index
        Returns: {assessment_id: relevance_score}
        """
        scores: Dict = {}
        query_lower = query.lower()
        
        # Split query into tokens
        tokens = query_lower.split()
        
        for token in tokens:
            # Direct token match
            if token in self.keyword_index:
                for assessment_id in self.keyword_index[token]:
                    scores[assessment_id] = scores.get(assessment_id, 0) + 1.0
            
            # Partial token match (substring)
            for keyword, assessment_ids in self.keyword_index.items():
                if token in keyword and len(token) > 3:  # Avoid short substrings
                    for assessment_id in assessment_ids:
                        scores[assessment_id] = scores.get(assessment_id, 0) + 0.5
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _semantic_search(self, query: str, k: int = 20) -> Dict[str, float]:
        """
        Semantic search using FAISS embeddings (optimized k=20 for speed)
        Returns: {assessment_id: similarity_score}
        """
        if self.faiss_index is None:
            return {}
        
        if not self._init_embedding_model():
            return {}
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)
        query_embedding = np.array([query_embedding])  # Make 2D for FAISS
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, min(k, self.faiss_index.ntotal))
        
        scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            assessment_id = self.reverse_id_mapping.get(int(idx))
            if assessment_id:
                # Convert L2 distance to similarity (smaller distance = higher similarity)
                similarity = 1 / (1 + dist)
                scores[assessment_id] = similarity
        
        return scores
    
    
    def _make_hashable(self, obj):
        """Convert unhashable types (lists, dicts) to hashable equivalents for caching"""
        if isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        else:
            return obj
    
    def search(
        self,
        query: str,
        constraints: Optional[Dict] = None,
        limit: int = 10,
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> List[Dict]:
        """
        Hybrid search combining keyword and semantic results (optimized weights for speed)
        
        Args:
            query: Search query string
            constraints: Optional constraints (job_levels, keys/categories, languages)
            limit: Maximum number of results
            keyword_weight: Weight for keyword search scores (0-1) — increased to 0.5 for faster path
            semantic_weight: Weight for semantic search scores (0-1) — reduced to 0.5 for speed
        
        Returns:
            List of assessment dictionaries, sorted by relevance
        """
        # Check cache first (convert constraints to hashable form)
        constraints_hashable = self._make_hashable(constraints) if constraints else None
        cache_key = (query, constraints_hashable, limit)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Get keyword scores
        keyword_scores = self._keyword_search(query)
        
        # Get semantic scores
        semantic_scores = self._semantic_search(query)
        
        # Combine scores
        combined_scores = {}
        for assessment_id in set(keyword_scores.keys()) | set(semantic_scores.keys()):
            kw_score = keyword_scores.get(assessment_id, 0)
            sem_score = semantic_scores.get(assessment_id, 0)
            combined = (keyword_weight * kw_score) + (semantic_weight * sem_score)
            combined_scores[assessment_id] = combined
        
        print(f"   [Retriever] Query: '{query[:50]}...'")
        print(f"   [Retriever] Combined scores before filtering: {len(combined_scores)} assessments")
        
        # Apply constraints
        if constraints:
            print(f"   [Retriever] Applying constraints: {list(constraints.keys())}")
            combined_scores = self._apply_constraints(combined_scores, constraints)
            print(f"   [Retriever] After constraint filtering: {len(combined_scores)} assessments")
        
        # Sort and limit
        sorted_ids = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # Build results with assessment data
        results = [
            self.assessments[assessment_id]
            for assessment_id, score in sorted_ids
            if assessment_id in self.assessments
        ]
        
        print(f"   [Retriever] Final results: {len(results)} assessments returned")
        
        # Cache result (simple LRU-like eviction)
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry (first item in dict since Python 3.7+ preserves insertion order)
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = results
        
        return results
    
    def _apply_constraints(self, scores: Dict[str, float], constraints: Dict) -> Dict[str, float]:
        """
        Filter scores based on constraints
        
        Constraints:
            - job_levels: List of required job levels
            - keys: List of required assessment categories
            - languages: List of required languages
            - test_types: List of required test types (cognitive, personality, situational_judgement, etc.)
        """
        # Map user test types to assessment keys (exact names from assessments.json)
        test_type_to_keys = {
            "cognitive": ["Ability & Aptitude"],
            "personality": ["Personality & Behavior"],
            "situational_judgement": ["Biodata & Situational Judgment"],
            "leadership": ["Competencies", "Assessment Exercises"],
            "skills": ["Knowledge & Skills"],
        }
        
        filtered_scores = {}
        
        for assessment_id, score in scores.items():
            assessment = self.assessments.get(assessment_id)
            if not assessment:
                continue
            
            # Check job levels
            if "job_levels" in constraints and constraints["job_levels"]:
                if not any(level in assessment.get("job_levels", []) 
                          for level in constraints["job_levels"]):
                    continue
            
            # Check assessment categories (keys)
            if "keys" in constraints and constraints["keys"]:
                if not any(key in assessment.get("keys", []) 
                          for key in constraints["keys"]):
                    continue
            
            # Check languages (case-insensitive partial match; skip if assessment has no languages specified)
            assessment_langs = assessment.get("languages", [])
            if "languages" in constraints and constraints["languages"] and assessment_langs:
                # Only apply language filter if assessment explicitly specifies languages
                constraint_langs_lower = [lang.lower() for lang in constraints["languages"]]
                if not any(any(constraint_lang in assessment_lang.lower() 
                              for constraint_lang in constraint_langs_lower)
                          for assessment_lang in assessment_langs):
                    continue
            
            # Check test types (map user test types to assessment keys)
            if "test_types" in constraints and constraints["test_types"]:
                required_keys = []
                for test_type in constraints["test_types"]:
                    if test_type in test_type_to_keys:
                        required_keys.extend(test_type_to_keys[test_type])
                
                # Assessment must have at least one matching key
                if required_keys:
                    assessment_keys = assessment.get("keys", [])
                    if not any(key in assessment_keys for key in required_keys):
                        continue
            
            filtered_scores[assessment_id] = score
        
        return filtered_scores
    
    def get_assessment_by_name(self, name: str) -> Optional[Dict]:
        """Get assessment by exact name (case-insensitive)"""
        name_lower = name.lower()
        for assessment in self.assessment_list:
            if assessment["name"].lower() == name_lower:
                return assessment
        return None
    
    def get_similar_assessments(self, assessment_name: str, k: int = 5) -> List[Dict]:
        """Get assessments similar to a given assessment name"""
        assessment = self.get_assessment_by_name(assessment_name)
        if not assessment:
            return []
        
        # Use the assessment's description as query
        query = f"{assessment['name']} {assessment['description']}"
        results = self.search(query, limit=k+1)  # +1 to exclude the original
        
        # Filter out the original assessment
        return [r for r in results if r["id"] != assessment["id"]][:k]


if __name__ == "__main__":
    # Test the retriever
    print("=" * 60)
    print("Testing HybridRetriever")
    print("=" * 60)
    
    try:
        retriever = HybridRetriever()
        
        # Test queries
        test_queries = [
            ("Java developer backend engineer", {}),
            ("leadership management", {"keys": ["Personality & Behavior"]}),
            ("numeric reasoning", {}),
        ]
        
        for query, constraints in test_queries:
            print(f"\n Query: '{query}'")
            if constraints:
                print(f"  Constraints: {constraints}")
            
            results = retriever.search(query, constraints=constraints, limit=5)
            print(f"  Found {len(results)} assessments:")
            for i, result in enumerate(results[:3], 1):
                print(f"    {i}. {result['name']}")
    
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
