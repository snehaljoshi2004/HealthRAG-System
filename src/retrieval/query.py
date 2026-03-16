# query.py
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareRAG:
    def __init__(self, persist_directory="./chroma_db"):
        # Use same embedding model as ingestion
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embeddings on {device}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load the existing vector store
        logger.info(f"Loading vector store from {persist_directory}...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        logger.info("✓ RAG system ready!")
    
    def retrieve(self, query: str, k: int = 5):
        """Retrieve top k relevant chunks for a query"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def format_results(self, results):
        """Format results for display"""
        formatted = []
        for i, (doc, score) in enumerate(results):
            formatted.append({
                "rank": i + 1,
                "score": round(score, 4),
                "source": doc.metadata.get('source_file', 'Unknown'),
                "content": doc.page_content.strip(),
                "preview": doc.page_content[:200] + "..."
            })
        return formatted
    
    def interactive_query(self):
        """Interactive query mode"""
        print("\n" + "="*60)
        print("HEALTHCARE RAG SYSTEM - READY")
        print("="*60)
        print("Type your questions (or 'quit' to exit)")
        print("-"*60)
        
        while True:
            query = input("\n🔍 Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n📚 Retrieving relevant information...")
            results = self.retrieve(query, k=3)
            
            print(f"\n📋 Found {len(results)} relevant passages:")
            for result in self.format_results(results):
                print(f"\n--- [{result['rank']}] Score: {result['score']} | Source: {result['source']} ---")
                print(result['preview'])

def main():
    rag = HealthcareRAG()
    
    # Example queries from your dataset
    example_queries = [
        "What is the significance of positive Babinski sign with muscle atrophy?",
        "What are the diagnostic criteria for diabetes?",
        "How does imatinib work in GIST patients?",
        "What are the symptoms of Lyme disease?",
        "What is the role of ACE inhibitors in heart failure?"
    ]
    
    print("\n📝 Example queries you can try:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    
    rag.interactive_query()

if __name__ == "__main__":
    main()