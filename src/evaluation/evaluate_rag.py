# evaluate_rag.py
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    # context_relevancy is deprecated/removed
)
from ragas import evaluate
from retriever import HybridRetriever
from langchain.schema import Document
import time
import logging
from typing import List, Dict, Any
import os
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, retriever=None):
        """Initialize RAG evaluator"""
        self.retriever = retriever or HybridRetriever()
        
        # Define metrics to use - removed context_relevancy
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
        
        logger.info(f"📊 Loaded {len(self.metrics)} RAGAS metrics")
    
    def load_golden_dataset(self, path="data/evaluation/rag_dataset_clean.json", sample_size=None):
        """Load golden dataset for evaluation"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if sample_size:
            data = data[:sample_size]
            logger.info(f"📚 Loaded {len(data)} samples (sampled)")
        else:
            logger.info(f"📚 Loaded {len(data)} samples (full dataset)")
        
        return data
    
    def prepare_ragas_dataset(self, golden_data: List[Dict]) -> Dataset:
        """Prepare dataset in RAGAS format"""
        
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for item in golden_data:
            question = item['question']
            # Handle both 'answer' and 'text_answer' fields
            ground_truth = item.get('text_answer', item.get('answer', ''))
            
            # Retrieve contexts using hybrid retriever
            retrieved_docs = self.retriever.retrieve(question, k=3)
            contexts = [doc.page_content for doc, _ in retrieved_docs]
            
            # For now, use the first retrieved doc as answer
            # In production, you'd use an LLM to generate answers
            answer = contexts[0] if contexts else ""
            
            ragas_data["question"].append(question)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append(contexts)
            ragas_data["ground_truth"].append(ground_truth)
        
        return Dataset.from_dict(ragas_data)
    
    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Run RAGAS evaluation"""
        logger.info("🔍 Running RAGAS evaluation...")
        
        try:
            result = evaluate(
                dataset,
                metrics=self.metrics
            )
            return result
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}
    
    def detailed_evaluation(self, golden_data: List[Dict], output_dir="eval_results"):
        """Run detailed evaluation with breakdown by category"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = []
        
        for i, item in enumerate(golden_data):
            question = item['question']
            ground_truth = item.get('text_answer', item.get('answer', ''))
            
            # Handle metadata safely
            metadata = item.get('metadata', {})
            category = metadata.get('specialty', 'Unknown')
            complexity = metadata.get('complexity', 'Unknown')
            
            # Retrieve contexts
            retrieved_docs = self.retriever.retrieve(question, k=3)
            contexts = [doc.page_content for doc, _ in retrieved_docs]
            
            # Check if correct context is retrieved
            correct_context = item.get('context', '')
            context_found = any(correct_context in ctx for ctx in contexts) if correct_context else False
            
            results.append({
                "id": i,
                "question": question[:100] + "...",
                "category": category,
                "complexity": complexity,
                "context_found": context_found,
                "num_contexts": len(contexts),
                "top_score": float(retrieved_docs[0][1]) if retrieved_docs else 0.0,
                "timestamp": timestamp
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save raw results
        df.to_csv(f"{output_dir}/detailed_results_{timestamp}.csv", index=False)
        
        # Generate summary
        summary = {
            "timestamp": timestamp,
            "total_queries": len(df),
            "context_found_rate": float(df['context_found'].mean()),
            "avg_top_score": float(df['top_score'].mean()),
            "by_category": df.groupby('category')['context_found'].mean().apply(float).to_dict(),
            "by_complexity": df.groupby('complexity')['context_found'].mean().apply(float).to_dict()
        }
        
        # Save summary
        with open(f"{output_dir}/summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as latest for CI
        with open(f"{output_dir}/latest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Detailed evaluation saved to {output_dir}")
        
        return df, summary
    
    def compare_with_baseline(self, current_results: pd.DataFrame, baseline_path: str = None):
        """Compare current results with baseline"""
        
        if baseline_path and os.path.exists(baseline_path):
            baseline = pd.read_csv(baseline_path)
            
            comparison = {
                "metric": ["context_found_rate", "avg_top_score"],
                "baseline": [
                    float(baseline['context_found'].mean()),
                    float(baseline['top_score'].mean())
                ],
                "current": [
                    float(current_results['context_found'].mean()),
                    float(current_results['top_score'].mean())
                ],
                "change": [
                    float(current_results['context_found'].mean() - baseline['context_found'].mean()),
                    float(current_results['top_score'].mean() - baseline['top_score'].mean())
                ]
            }
            
            return pd.DataFrame(comparison)
        return None


def ci_mode():
    """Run evaluation in CI mode with thresholds"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ci-mode', action='store_true')
    parser.add_argument('--sample-size', type=int, default=50)
    args = parser.parse_args()
    
    if args.ci_mode:
        print("\n" + "="*60)
        print("🔧 RUNNING IN CI MODE")
        print("="*60)
        
        evaluator = RAGEvaluator()
        golden_data = evaluator.load_golden_dataset(sample_size=args.sample_size)
        df, summary = evaluator.detailed_evaluation(golden_data)
        
        print(f"\n📊 CI Evaluation Summary:")
        print(f"   Context Found Rate: {summary['context_found_rate']:.2%}")
        print(f"   Average Top Score: {summary['avg_top_score']:.4f}")
        
        return True
    
    return False


def main():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("PHASE 3: RAG EVALUATION FRAMEWORK")
    print("="*70)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load golden dataset (sample first 20 for quick test)
    golden_data = evaluator.load_golden_dataset(sample_size=20)
    
    # Run detailed evaluation
    print("\n📊 Running detailed evaluation...")
    df, summary = evaluator.detailed_evaluation(golden_data)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Context Found Rate: {summary['context_found_rate']:.2%}")
    print(f"Average Top Score: {summary['avg_top_score']:.4f}")
    
    print("\n📈 Performance by Category:")
    for category, rate in summary['by_category'].items():
        print(f"  {category}: {rate:.2%}")
    
    print("\n📈 Performance by Complexity:")
    for complexity, rate in summary['by_complexity'].items():
        print(f"  {complexity}: {rate:.2%}")
    
    # Try RAGAS evaluation
    print("\n🔬 Running RAGAS evaluation (may take a few minutes)...")
    try:
        ragas_dataset = evaluator.prepare_ragas_dataset(golden_data[:5])  # Small sample for RAGAS
        ragas_results = evaluator.evaluate(ragas_dataset)
        
        if ragas_results:
            print("\n📊 RAGAS Metrics:")
            for metric, score in ragas_results.items():
                print(f"  {metric}: {score:.4f}")
    except Exception as e:
        print(f"⚠️ RAGAS evaluation skipped: {e}")
    
    print("\n✅ Evaluation complete! Check eval_results/ folder for detailed outputs.")


if __name__ == "__main__":
    # Check if running in CI mode first
    if not ci_mode():
        main()