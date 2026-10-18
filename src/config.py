"""Configuration and Environment Settings for LedgerGone."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# Database Configuration (TigerData TimescaleDB)
DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

# Ollama / Gemma Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma3")
EMBEDDING_DIM = 384  # Matches BAAI/bge-small-en-v1.5 ONNX CPU embeddings
