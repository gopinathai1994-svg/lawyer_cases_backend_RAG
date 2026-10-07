import os
from pathlib import Path
from dotenv import load_dotenv

# Load values from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# User documents go only inside this folder.
DOCUMENTS_DIR = BASE_DIR / "documents"

# Chroma will save the vector database here.
VECTOR_DB_DIR = BASE_DIR / "vector_db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Free-tier friendly model choices.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)

GEMINI_EMBED_MODEL = os.getenv(
    "GEMINI_EMBED_MODEL",
    "gemini-embedding-001"
)

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))

# We report Gemini API cost as zero for your free-tier project.
# You can change these later if you move to a paid plan.
GEMINI_INPUT_PRICE_PER_1M = float(
    os.getenv("GEMINI_INPUT_PRICE_PER_1M", "0")
)
GEMINI_OUTPUT_PRICE_PER_1M = float(
    os.getenv("GEMINI_OUTPUT_PRICE_PER_1M", "0")
)

# BERTScore language. Change to another supported language if needed.
BERTSCORE_LANG = os.getenv("BERTSCORE_LANG", "en")
