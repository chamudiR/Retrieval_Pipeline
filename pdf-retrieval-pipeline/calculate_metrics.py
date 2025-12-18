import json

def calculate_metrics(results_file='evaluation_results.json'):
    """Calculate retrieval evaluation metrics"""
    
    # Load annotated results
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ {results_file} not found. Run evaluate_retrieval.py first.")
        return
    
    print(f"\n{'='*80}")
    print("RETRIEVAL EVALUATION METRICS")
    print(f"{'='*80}\n")
    
    total_questions = len(results)
    k_values = [1, 3, 5]  # Calculate metrics at different cutoffs
    
    metrics = {
        'precision_at_k': {k: [] for k in k_values},
        'recall_at_k': {k: [] for k in k_values},
        'mrr': [],  # Mean Reciprocal Rank
        'success_at_k': {k: [] for k in k_values}  # Found at least 1 relevant
    }
    
    for question_result in results:
        q_id = question_result['question_id']
        chunks = question_result['retrieved_chunks']
        
        # Check if any chunks are annotated
        annotated = any(chunk.get('relevant') is not None for chunk in chunks)
        if not annotated:
            print(f"⚠️  Question {q_id} not annotated yet (skipping)")
            continue
        
        # Get relevant chunks
        relevant_ranks = [chunk['rank'] for chunk in chunks if chunk.get('relevant') == True]
        total_relevant = len(relevant_ranks)
        
        # Calculate MRR (Mean Reciprocal Rank)
        if relevant_ranks:
            first_relevant_rank = min(relevant_ranks)
            mrr = 1.0 / first_relevant_rank
            metrics['mrr'].append(mrr)
        else:
            metrics['mrr'].append(0)
        
        # Calculate metrics at different K values
        for k in k_values:
            chunks_at_k = chunks[:k]
            relevant_at_k = [c for c in chunks_at_k if c.get('relevant') == True]
            
            # Precision@K = relevant retrieved / k
            precision = len(relevant_at_k) / k
            metrics['precision_at_k'][k].append(precision)
            
            # Recall@K = relevant retrieved / total relevant
            # (Assumes we don't know total relevant in corpus, just in top K)
            if total_relevant > 0:
                recall = len(relevant_at_k) / total_relevant
            else:
                recall = 0
            metrics['recall_at_k'][k].append(recall)
            
            # Success@K = found at least 1 relevant
            success = 1 if len(relevant_at_k) > 0 else 0
            metrics['success_at_k'][k].append(success)
        
        # Print per-question results
        print(f"Q{q_id}: Relevant={total_relevant}, First@{min(relevant_ranks) if relevant_ranks else 'N/A'}")
    
    # Calculate averages
    print(f"\n{'='*80}")
    print("AVERAGE METRICS")
    print(f"{'='*80}\n")
    
    if not metrics['mrr']:
        print("❌ No annotated questions found. Please annotate evaluation_results.json first.")
        return
    
    evaluated_questions = len(metrics['mrr'])
    print(f"📊 Evaluated Questions: {evaluated_questions}/{total_questions}\n")
    
    # MRR
    avg_mrr = sum(metrics['mrr']) / len(metrics['mrr'])
    print(f"🎯 Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")
    print(f"   (Higher is better, 1.0 = perfect)\n")
    
    # Precision, Recall, Success at K
    for k in k_values:
        print(f"📈 Metrics @ Top-{k}:")
        
        avg_precision = sum(metrics['precision_at_k'][k]) / len(metrics['precision_at_k'][k])
        print(f"   • Precision@{k}: {avg_precision:.3f}")
        
        avg_recall = sum(metrics['recall_at_k'][k]) / len(metrics['recall_at_k'][k])
        print(f"   • Recall@{k}: {avg_recall:.3f}")
        
        avg_success = sum(metrics['success_at_k'][k]) / len(metrics['success_at_k'][k])
        print(f"   • Success@{k}: {avg_success:.3f} ({int(avg_success*100)}% questions)")
        print()
    
    print(f"{'='*80}")
    print("\n📝 Interpretation:")
    print("  • Precision@K: What % of retrieved chunks are relevant")
    print("  • Recall@K: What % of relevant chunks were retrieved")
    print("  • MRR: How high is the first relevant result ranked")
    print("  • Success@K: % of questions with at least 1 relevant result in top K")
    print(f"{'='*80}\n")
    
    # Save metrics summary
    summary = {
        "evaluated_questions": evaluated_questions,
        "total_questions": total_questions,
        "mrr": avg_mrr,
        "precision_at_k": {k: sum(metrics['precision_at_k'][k]) / len(metrics['precision_at_k'][k]) for k in k_values},
        "recall_at_k": {k: sum(metrics['recall_at_k'][k]) / len(metrics['recall_at_k'][k]) for k in k_values},
        "success_at_k": {k: sum(metrics['success_at_k'][k]) / len(metrics['success_at_k'][k]) for k in k_values}
    }
    
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("💾 Metrics saved to: evaluation_metrics.json\n")

if __name__ == "__main__":
    calculate_metrics()
