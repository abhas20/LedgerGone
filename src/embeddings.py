"""CPU-optimized Vector Embeddings Engine for LedgerGone using FastEmbed (ONNX)."""
from typing import List, Union
from fastembed import TextEmbedding
from config import EMBEDDING_DIM, MODEL_CACHE_DIR
import math

_embedding_model = None

def _unit_fallback_vector() -> List[float]:
    """Return a unit-length fallback vector so pgvector cosine distance (<=>) never hits division-by-zero NaN."""
    val = 1.0 / math.sqrt(EMBEDDING_DIM)
    return [val] * EMBEDDING_DIM


def get_embedding_model() -> TextEmbedding:
    """Singleton getter for the ONNX FastEmbed model (runs purely on CPU)."""
    global _embedding_model
    if _embedding_model is None:
        # Saves ONNX model in project .model_cache instead of volatile Linux /tmp
        _embedding_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(MODEL_CACHE_DIR)
        )
    return _embedding_model

def generate_embedding(text: str) -> List[float]:
    """Generate a 384-dimensional vector embedding for a single string."""
    cleaned = (text or "").strip()
    if not cleaned:
        return _unit_fallback_vector()
    try:
        model = get_embedding_model()
        embeddings = list(model.embed([cleaned]))
        return embeddings[0].tolist()
    except Exception as e:
        print(f"[FastEmbed Notice: {e}] Using unit fallback vector.")
        return _unit_fallback_vector()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch generate 384-dimensional vector embeddings."""
    if not texts:
        return []
    cleaned_texts = [(t or "").strip() or "general kirana item" for t in texts]
    try:
        model = get_embedding_model()
        embeddings = list(model.embed(cleaned_texts))
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"[FastEmbed Batch Notice: {e}] Using unit fallback vectors.")
        return [_unit_fallback_vector() for _ in texts]