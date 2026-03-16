# hybrid_retriever.py
import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document  # Add this import
from sentence_transformers import CrossEncoder
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, persist_directory="./chroma_db", bm25_index_path="./bm25_index.pkl"):
        self.persist_directory = persist_directory
        self.bm25_index_path = bm25_index_path
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load embeddings (same as before)
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load vector store
        logger.info("Loading vector store...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Load or create BM25 index
        self.chunks, self.bm25 = self._load_or_create_bm25()
        
        # Load cross-encoder for reranking
        logger.info("Loading cross-encoder reranker...")
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device=self.device
        )
        
        logger.info("✓ Hybrid retriever ready!")
    
    def _load_or_create_bm25(self) -> Tuple[List[Dict], BM25Okapi]:
        """Load existing BM25 index or create a new one"""
        
        # Check if BM25 index exists
        if os.path.exists(self.bm25_index_path):
            logger.info("Loading existing BM25 index...")
            with open(self.bm25_index_path, 'rb') as f:
                return pickle.load(f)
        
        # Create new BM25 index
        logger.info("Creating BM25 index from chunks...")
        
        # Get all chunks from vectorstore
        all_docs = self.vectorstore.get()
        
        # Extract text content
        chunks = []
        for i, text in enumerate(all_docs['documents']):
            chunks.append({
                'id': i,
                'text': text,
                'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {}
            })
        
        # Prepare tokenized corpus for BM25
        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in chunks]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save for future use
        logger.info(f"Saving BM25 index to {self.bm25_index_path}")
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump((chunks, bm25), f)
        
        return chunks, bm25
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25"""
        return text.lower().split()
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining:
        - Vector search (semantic)
        - BM25 search (keyword)
        """
        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # BM25 search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[-k*2:][::-1]
        
        # Combine results
        all_candidates = []
        seen_texts = set()
        
        # Add vector results
        for doc, score in vector_results:
            text = doc.page_content
            if text not in seen_texts:
                seen_texts.add(text)
                all_candidates.append({
                    'document': doc,
                    'vector_score': score,
                    'bm25_score': 0.0,
                    'text': text
                })
        
        # Add BM25 results (avoid duplicates)
        for idx in bm25_top_indices:
            text = self.chunks[idx]['text']
            if text not in seen_texts:
                seen_texts.add(text)
                doc = Document(
                    page_content=text,
                    metadata=self.chunks[idx]['metadata']
                )
                all_candidates.append({
                    'document': doc,
                    'vector_score': 0.0,
                    'bm25_score': bm25_scores[idx],
                    'text': text
                })
        
        return all_candidates
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Rerank candidates using cross-encoder
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, cand['text']] for cand in candidates]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Combine scores
        final_scores = []
        for i, cand in enumerate(candidates):
            # Normalize scores (lower vector_score is better, so we invert)
            vector_score = 1.0 / (1.0 + cand.get('vector_score', 100.0))
            
            # Normalize BM25 score (higher is better)
            max_bm25 = max([c.get('bm25_score', 0) for c in candidates]) or 1.0
            bm25_score = cand.get('bm25_score', 0) / max_bm25
            
            # Cross-encoder score (higher is better)
            cross_score = (cross_scores[i] + 1) / 2  # Normalize to 0-1 range if needed
            
            # Weighted combination (tune these weights)
            combined_score = (
                0.15 * vector_score +
                0.15 * bm25_score +
                0.7 * cross_score
            )
            
            final_scores.append((cand['document'], combined_score))
        
        # Sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores[:top_k]
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Complete retrieval pipeline:
        1. Hybrid search to get candidates
        2. Rerank candidates with cross-encoder
        """
        # Step 1: Hybrid search
        candidates = self.hybrid_search(query, k=k*2)
        
        # Step 2: Rerank
        reranked_results = self.rerank(query, candidates, top_k=k)
        
        return reranked_results

# For testing
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    # Test with sample queries
    test_queries = [
        "What is the significance of positive Babinski sign with muscle atrophy?",
        "How does imatinib work in GIST patients?",
        "What are the diagnostic criteria for diabetes?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = retriever.retrieve(query, k=3)
        
        for i, (doc, score) in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Preview: {doc.page_content[:200]}...")