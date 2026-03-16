# query_v2.py
from retriever import HybridRetriever
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareRAGv2:
    def __init__(self):
        logger.info("Initializing Phase 2 RAG system...")
        self.retriever = HybridRetriever()
    
    def answer(self, query: str, k: int = 5):
        """Retrieve relevant chunks for a query"""
        results = self.retriever.retrieve(query, k=k)
        return results
    
    def interactive(self):
        """Interactive query mode"""
        print("\n" + "="*60)
        print("HEALTHCARE RAG SYSTEM v2 - WITH HYBRID SEARCH + RERANKING")
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
            
            print("\n📚 Retrieving and reranking...")
            results = self.retriever.retrieve(query, k=3)
            
            print(f"\n📋 Found {len(results)} relevant passages:")
            for i, (doc, score) in enumerate(results):
                print(f"\n--- [{i+1}] Score: {score:.4f} | Source: {doc.metadata.get('source_file', 'Unknown')} ---")
                print(doc.page_content[:300] + "...")

def main():
    rag = HealthcareRAGv2()
    rag.interactive()

if __name__ == "__main__":
    main()