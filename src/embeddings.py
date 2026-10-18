"""CPU-optimized Vector Embeddings Engine for LedgerGone using FastEmbed (ONNX)."""
from typing import List, Union
from fastembed import TextEmbedding
from src.config import EMBEDDING_DIM

_embedding_model = None

def get_embedding_model() -> TextEmbedding:
    """Singleton getter for the ONNX FastEmbed model (runs purely on CPU)."""
    global _embedding_model
    if _embedding_model is None:
        # BAAI/bge-small-en-v1.5 generates 384-dimensional embeddings on CPU in <10ms
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model

def generate_embedding(text: str) -> List[float]:
    """Generate a 384-dimensional vector embedding for a single string."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    model = get_embedding_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch generate 384-dimensional vector embeddings."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]
