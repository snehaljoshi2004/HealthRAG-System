# check_thresholds.py
import json
import sys
import argparse

def check_thresholds(context_rate_threshold=0.7, score_threshold=1.5):
    """Check if evaluation results meet thresholds"""
    
    # Find latest summary
    import glob
    import os
    
    summary_files = glob.glob("eval_results/summary_*.json")
    if not summary_files:
        print("❌ No evaluation results found")
        sys.exit(1)
    
    latest_file = max(summary_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        summary = json.load(f)
    
    print(f"\n📊 Checking thresholds against {latest_file}")
    print(f"   Context Found Rate: {summary['context_found_rate']:.2%} (threshold: {context_rate_threshold:.0%})")
    print(f"   Average Top Score: {summary['avg_top_score']:.4f} (threshold: {score_threshold})")
    
    passed = True
    
    if summary['context_found_rate'] < context_rate_threshold:
        print(f"❌ Context found rate below threshold")
        passed = False
    
    if summary['avg_top_score'] < score_threshold:
        print(f"❌ Average top score below threshold")
        passed = False
    
    if passed:
        print("✅ All thresholds passed!")
        sys.exit(0)
    else:
        print("❌ Threshold check failed")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-rate', type=float, default=0.7)
    parser.add_argument('--avg-score', type=float, default=1.5)
    args = parser.parse_args()
    
    check_thresholds(args.context_rate, args.avg_score)