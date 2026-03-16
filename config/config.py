# config.py
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    EVAL_DIR = DATA_DIR / "evaluation"
    VECTORSTORE_DIR = BASE_DIR / "chroma_db"
    BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"
    
    # Model settings
    EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    # Retrieval
    TOP_K = 5
    RERANK_TOP_K = 3
    
    # Evaluation
    CONTEXT_FOUND_THRESHOLD = 0.5
    AVG_SCORE_THRESHOLD = 1.5