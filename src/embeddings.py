"""
Task 1.3 (Updated): Generate Embeddings with FAISS
Generates embeddings and indexes them using FAISS for fast similarity search
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys


def generate_embeddings_with_faiss(
    assessments_path: str = "data/assessments.json",
    faiss_index_path: str = "data/assessments_index.faiss",
    id_mapping_path: str = "data/id_mapping.json",
    model_name: str = "all-MiniLM-L6-v2",
):
    """
    Generate embeddings for all assessments and index with FAISS
    
    Args:
        assessments_path: Path to normalized assessments.json
        faiss_index_path: Path to save FAISS index
        id_mapping_path: Path to save id mapping (assessment_id -> index_position)
        model_name: HuggingFace model identifier for Sentence-Transformers
    """
    
    print("=" * 60)
    print("Task 1.3: Generate Embeddings with FAISS")
    print("=" * 60)
    
    # Import dependencies
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        missing = []
        try:
            from sentence_transformers import SentenceTransformer
        except:
            missing.append("sentence-transformers")
        try:
            import faiss
        except:
            missing.append("faiss-cpu")
        print(f" Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        sys.exit(1)
    
    # Load assessments
    print(f"\n1. Loading assessments from {assessments_path}...")
    with open(assessments_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)
    print(f"    Loaded {len(assessments)} assessments")
    
    # Load model
    print(f"\n2. Loading embedding model '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_embedding_dimension()
        print(f"    Model loaded (dimension: {embedding_dim})")
    except Exception as e:
        print(f"    Failed to load model: {e}")
        sys.exit(1)
    
    # Generate embeddings and build FAISS index
    print(f"\n3. Generating embeddings and building FAISS index...")
    embeddings_list = []
    id_mapping = {}  # assessment_id -> index_position
    
    for idx, assessment in enumerate(assessments):
        if (idx + 1) % 50 == 0:
            print(f"   Processing {idx + 1}/{len(assessments)}...")
        
        # Concatenate name and description for embedding
        text = f"{assessment['name']} {assessment['description']}"
        
        # Truncate to ~2000 chars
        if len(text) > 2000:
            text = text[:2000]
        
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        embeddings_list.append(embedding)
        
        # Store mapping
        id_mapping[assessment["id"]] = idx
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list).astype(np.float32)
    print(f"    Generated {len(embeddings_list)} embeddings")
    print(f"    Embedding shape: {embeddings_array.shape}")
    
    # Create FAISS index (using IndexFlatL2 for simplicity, suitable for ~400 items)
    print(f"\n4. Creating FAISS index...")
    try:
        # IndexFlatL2: Exhaustive search using L2 distance (suitable for smaller datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSW
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_array)
        print(f"    FAISS index created (IndexFlatL2, {index.ntotal} vectors)")
    except Exception as e:
        print(f"    Failed to create FAISS index: {e}")
        sys.exit(1)
    
    # Save FAISS index
    print(f"\n5. Saving FAISS index to {faiss_index_path}...")
    output_dir = Path(faiss_index_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        faiss.write_index(index, faiss_index_path)
        print(f"    FAISS index saved ({Path(faiss_index_path).stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"    Failed to save FAISS index: {e}")
        sys.exit(1)
    
    # Save ID mapping
    print(f"\n6. Saving ID mapping to {id_mapping_path}...")
    with open(id_mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2)
    print(f"    ID mapping saved ({len(id_mapping)} assessments)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total embeddings: {len(embeddings_list)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Model: {model_name}")
    print(f"  Index type: IndexFlatL2")
    print(f"  FAISS index size: {Path(faiss_index_path).stat().st_size / (1024*1024):.2f} MB")
    print("=" * 60)
    
    return index, embeddings_array, id_mapping, assessments


def test_faiss_retrieval(index_path: str = "data/assessments_index.faiss",
                         id_mapping_path: str = "data/id_mapping.json",
                         assessments_path: str = "data/assessments.json",
                         k: int = 3):
    """Test FAISS retrieval with embedding-based queries"""
    print("\n" + "=" * 60)
    print("Testing FAISS Retrieval")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print(" sentence-transformers or faiss-cpu not available")
        return
    
    # Load index
    index = faiss.read_index(index_path)
    print(f"\n Loaded FAISS index ({index.ntotal} vectors)")
    
    # Load ID mapping
    with open(id_mapping_path, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)
    
    # Create reverse mapping (index_position -> assessment_id)
    reverse_mapping = {v: k for k, v in id_mapping.items()}
    
    # Load assessments
    with open(assessments_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)
    assessment_by_id = {a["id"]: a for a in assessments}
    
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test queries
    test_queries = [
        "Java developer",
        "leadership assessment",
        "numeric reasoning",
    ]
    
    for query in test_queries:
        # Encode query
        query_embedding = model.encode(query, convert_to_numpy=True).astype(np.float32)
        query_embedding = np.array([query_embedding])  # Make it 2D for FAISS
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        print(f"\n Query: '{query}'")
        print(f"   Top {k} similar assessments:")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            assessment_id = reverse_mapping[idx]
            assessment = assessment_by_id[assessment_id]
            # Convert L2 distance to similarity score (approximate)
            similarity = 1 / (1 + dist)
            print(f"      {rank}. {assessment['name']} (distance: {dist:.3f}, similarity: {similarity:.3f})")


if __name__ == "__main__":
    generate_embeddings_with_faiss()
    test_faiss_retrieval()
