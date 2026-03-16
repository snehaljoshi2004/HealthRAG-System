# compare_ret.py
import json
from src.retrieval.query import HealthcareRAG
from q2 import HealthcareRAGv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_on_golden_dataset():
    # Load golden dataset
    with open('data/evaluation/rag_dataset_clean.json', 'r') as f:
        dataset = json.load(f)
    
    # Initialize both retrievers
    print("Initializing Phase 1 retriever...")
    rag_v1 = HealthcareRAG()
    
    print("\nInitializing Phase 2 retriever...")
    rag_v2 = HealthcareRAGv2()
    
    # Test first 10 questions
    print("\n" + "="*70)
    print("COMPARING PHASE 1 VS PHASE 2 RETRIEVAL")
    print("="*70)
    
    results_comparison = []
    
    for i, item in enumerate(dataset[:10]):
        question = item['question']
        expected_context = item['context']
        
        print(f"\n{i+1}. Question: {question[:100]}...")
        
        # Phase 1 retrieval
        start = time.time()
        results_v1 = rag_v1.retrieve(question, k=3)  # Fixed: using retrieve() directly
        time_v1 = time.time() - start
        
        # Phase 2 retrieval
        start = time.time()
        results_v2 = rag_v2.answer(question, k=3)  # Using answer() method from q2.py
        time_v2 = time.time() - start
        
        # Check if expected context found
        found_v1 = False
        found_v2 = False
        
        # Check Phase 1 results
        for doc, score in results_v1:
            if expected_context.strip() in doc.page_content.strip():
                found_v1 = True
                top_score_v1 = score
                break
        else:
            top_score_v1 = results_v1[0][1] if results_v1 else 0
        
        # Check Phase 2 results
        for doc, score in results_v2:
            if expected_context.strip() in doc.page_content.strip():
                found_v2 = True
                top_score_v2 = score
                break
        else:
            top_score_v2 = results_v2[0][1] if results_v2 else 0
        
        print(f"   Phase 1: Found={found_v1}, Top Score={top_score_v1:.4f}, Time={time_v1:.2f}s")
        print(f"   Phase 2: Found={found_v2}, Top Score={top_score_v2:.4f}, Time={time_v2:.2f}s")
        
        results_comparison.append({
            "question": question[:50],
            "phase1_found": found_v1,
            "phase1_score": top_score_v1,
            "phase1_time": time_v1,
            "phase2_found": found_v2,
            "phase2_score": top_score_v2,
            "phase2_time": time_v2
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    v1_success = sum(1 for r in results_comparison if r['phase1_found'])
    v2_success = sum(1 for r in results_comparison if r['phase2_found'])
    
    print(f"Phase 1 Success Rate: {v1_success}/{len(results_comparison)} ({v1_success/len(results_comparison)*100:.1f}%)")
    print(f"Phase 2 Success Rate: {v2_success}/{len(results_comparison)} ({v2_success/len(results_comparison)*100:.1f}%)")
    
    avg_time_v1 = sum(r['phase1_time'] for r in results_comparison) / len(results_comparison)
    avg_time_v2 = sum(r['phase2_time'] for r in results_comparison) / len(results_comparison)
    
    print(f"Average Time - Phase 1: {avg_time_v1:.3f}s, Phase 2: {avg_time_v2:.3f}s")
    
    # Score comparison
    avg_score_v1 = sum(r['phase1_score'] for r in results_comparison) / len(results_comparison)
    avg_score_v2 = sum(r['phase2_score'] for r in results_comparison) / len(results_comparison)
    
    print(f"Average Top Score - Phase 1: {avg_score_v1:.4f}, Phase 2: {avg_score_v2:.4f}")

if __name__ == "__main__":
    compare_on_golden_dataset()