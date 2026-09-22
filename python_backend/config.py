"""Central configuration settings for Qdrant vector database, directories, model parameters, and RRF constants."""
import os
from pydantic_settings import BaseSettings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):

    # =========================
    # FILE / DATA DIRECTORIES
    # =========================

    UPLOAD_DIR: str = os.path.join(
        BASE_DIR,
        "uploaded_pdfs"
    )

    DATA_DIR: str = os.path.join(
        BASE_DIR,
        "data"
    )

    TERMINOLOGY_STORE_PATH: str = os.path.join(
        BASE_DIR,
        "data",
        "terminology_store.json"
    )

    # =========================
    # QDRANT VECTOR DB
    # =========================

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_INDEX_COLLECTION_NAME: str = "rajbhasha_index_collection"
    QDRANT_CONTENT_COLLECTION_NAME: str = "rajbhasha_content_collection"
    QDRANT_TABLE_COLLECTION_NAME: str = "rajbhasha_table_collection"
    QDRANT_HIERARCHICAL_COLLECTION_NAME: str = "rajbhasha_hierarchical_collection"

    EMBEDDING_MODEL_NAME: str = (
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    GENERATOR_MODEL_NAME: str = "llama3.2"

    # =========================
    # HYBRID RETRIEVAL (DENSE + SPARSE/BM25 + RRF)
    # =========================
    RRF_K: int = 60
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RETRIEVAL_TOP_K: int = 20

    # =========================
    # OLLAMA
    # =========================

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"

    # =========================
    # SETTINGS
    # =========================

    class Config:
        extra = "allow"


settings = Settings()
