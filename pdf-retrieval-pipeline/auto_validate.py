import json
import re
from sentence_transformers import SentenceTransformer, util

# Load embedding model once (same model used for retrieval)
print("Loading embedding model for semantic similarity...")
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print("✓ Model loaded\n")

def normalize_text(text):
    """Normalize text for comparison - remove extra spaces, lowercase"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip().lower()
    return text

def check_relevance_lexical(retrieved_content, correct_content, threshold=0.5):
    """
    Lexical matching: Check if retrieved chunk contains the correct answer
    Returns True if significant overlap found
    """
    retrieved_norm = normalize_text(retrieved_content)
    correct_norm = normalize_text(correct_content)
    
    # Check if correct content is substring of retrieved (allowing for some variation)
    # Split into meaningful phrases and check overlap
    correct_phrases = [p.strip() for p in correct_norm.split('.') if len(p.strip()) > 20]
    
    if not correct_phrases:
        # If correct answer is short, check direct containment
        return correct_norm in retrieved_norm
    
    # Count how many key phrases appear in retrieved chunk
    matches = sum(1 for phrase in correct_phrases if phrase in retrieved_norm)
    overlap_ratio = matches / len(correct_phrases)
    
    return overlap_ratio >= threshold

def check_relevance_semantic(retrieved_content, correct_content, threshold=0.75):
    """
    Semantic matching: Use embeddings to measure semantic similarity
    Returns True if cosine similarity exceeds threshold
    """
    # Encode both texts
    retrieved_emb = embedding_model.encode(retrieved_content, convert_to_tensor=True)
    correct_emb = embedding_model.encode(correct_content, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity = util.cos_sim(retrieved_emb, correct_emb).item()
    
    return similarity >= threshold

def check_relevance(retrieved_content, correct_content, method='semantic', semantic_threshold=0.75, lexical_threshold=0.5):
    """
    Check if retrieved chunk is relevant to the correct answer
    
    Args:
        retrieved_content: Text from retrieved chunk
        correct_content: Expected answer text
        method: 'semantic', 'lexical', or 'hybrid'
        semantic_threshold: Similarity threshold for semantic matching (0.0-1.0)
        lexical_threshold: Overlap threshold for lexical matching (0.0-1.0)
    
    Returns:
        bool: True if relevant, False otherwise
    """
    if method == 'semantic':
        return check_relevance_semantic(retrieved_content, correct_content, semantic_threshold)
    
    elif method == 'lexical':
        return check_relevance_lexical(retrieved_content, correct_content, lexical_threshold)
    
    elif method == 'hybrid':
        # Both methods must agree (strict) or either passes (lenient)
        semantic_match = check_relevance_semantic(retrieved_content, correct_content, semantic_threshold)
        lexical_match = check_relevance_lexical(retrieved_content, correct_content, lexical_threshold)
        # Lenient: at least one method says it's relevant
        return semantic_match or lexical_match
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'semantic', 'lexical', or 'hybrid'")

def auto_validate_results(evaluation_file='evaluation_results.json', qbank_file='Qbank.json', method='semantic', semantic_threshold=0.75, lexical_threshold=0.5):
    """
    Automatically validate evaluation results against Qbank
    
    Args:
        evaluation_file: Path to evaluation results JSON
        qbank_file: Path to Qbank JSON with expected answers
        method: 'semantic', 'lexical', or 'hybrid'
        semantic_threshold: Similarity threshold for semantic matching (default: 0.75)
        lexical_threshold: Overlap threshold for lexical matching (default: 0.5)
    """
    print(f"\n{'='*80}")
    print("AUTOMATIC VALIDATION")
    print(f"Method: {method.upper()}")
    if method == 'semantic':
        print(f"Semantic Similarity Threshold: {semantic_threshold}")
    elif method == 'lexical':
        print(f"Lexical Overlap Threshold: {lexical_threshold}")
    else:
        print(f"Semantic Threshold: {semantic_threshold}, Lexical Threshold: {lexical_threshold}")
    print(f"{'='*80}\n")
    
    # Load files
    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ {evaluation_file} not found. Run evaluate.py first.")
        return
    
    try:
        with open(qbank_file, 'r', encoding='utf-8') as f:
            qbank = json.load(f)
    except FileNotFoundError:
        print(f"❌ {qbank_file} not found.")
        return
    
    # Create qbank lookup by question_id
    qbank_lookup = {q['question_id']: q for q in qbank}
    
    validated_count = 0
    total_chunks = 0
    relevant_found = 0
    
    # Validate each question's results
    for eval_q in eval_results:
        q_id = eval_q['question_id']
        
        if q_id not in qbank_lookup:
            print(f"⚠️  Question {q_id}: No answer in Qbank (skipping)")
            continue
        
        correct_chunks = qbank_lookup[q_id]['retrieved_chunks']
        retrieved_chunks = eval_q['retrieved_chunks']
        
        print(f"\n📌 Question {q_id}:")
        
        # For each retrieved chunk, check if it matches any correct answer
        for retr_chunk in retrieved_chunks:
            total_chunks += 1
            is_relevant = False
            
            # Compare against all correct answer chunks
            for correct_chunk in correct_chunks:
                if check_relevance(retr_chunk['content'], correct_chunk['content'], 
                                 method=method, semantic_threshold=semantic_threshold, 
                                 lexical_threshold=lexical_threshold):
                    is_relevant = True
                    break
            
            # Update relevance
            retr_chunk['relevant'] = is_relevant
            
            if is_relevant:
                relevant_found += 1
                print(f"   ✓ Rank {retr_chunk['rank']}: RELEVANT - {retr_chunk['source_file']}")
            else:
                print(f"   ✗ Rank {retr_chunk['rank']}: Not relevant - {retr_chunk['source_file']}")
        
        validated_count += 1
    
    # Save updated evaluation results
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Questions validated: {validated_count}")
    print(f"📊 Total chunks checked: {total_chunks}")
    print(f"✓  Relevant chunks found: {relevant_found} ({relevant_found/total_chunks*100:.1f}%)")
    print(f"\n💾 Updated {evaluation_file} with relevance judgments")
    print(f"{'='*80}\n")
    
    # Calculate metrics immediately
    calculate_metrics(evaluation_file)

def calculate_metrics(results_file='evaluation_results.json'):
    """Calculate retrieval evaluation metrics"""
    
    # Load annotated results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"\n{'='*80}")
    print("RETRIEVAL EVALUATION METRICS")
    print(f"{'='*80}\n")
    
    total_questions = len(results)
    k_values = [1, 3, 5]
    
    metrics = {
        'precision_at_k': {k: [] for k in k_values},
        'recall_at_k': {k: [] for k in k_values},
        'mrr': [],
        'success_at_k': {k: [] for k in k_values}
    }
    
    for question_result in results:
        q_id = question_result['question_id']
        chunks = question_result['retrieved_chunks']
        
        # Get relevant chunks
        relevant_ranks = [chunk['rank'] for chunk in chunks if chunk.get('relevant') == True]
        total_relevant = len(relevant_ranks)
        
        # Calculate MRR
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
            
            # Precision@K
            precision = len(relevant_at_k) / k
            metrics['precision_at_k'][k].append(precision)
            
            # Recall@K
            if total_relevant > 0:
                recall = len(relevant_at_k) / total_relevant
            else:
                recall = 0
            metrics['recall_at_k'][k].append(recall)
            
            # Success@K
            success = 1 if len(relevant_at_k) > 0 else 0
            metrics['success_at_k'][k].append(success)
        
        # Print per-question results
        print(f"Q{q_id}: Relevant={total_relevant}, First@{min(relevant_ranks) if relevant_ranks else 'N/A'}")
    
    # Calculate averages
    print(f"\n{'='*80}")
    print("AVERAGE METRICS")
    print(f"{'='*80}\n")
    
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
    import sys
    
    # Parse command line arguments for method selection
    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method not in ['semantic', 'lexical', 'hybrid']:
            print("Usage: python auto_validate.py [semantic|lexical|hybrid]")
            print("Default: semantic")
            sys.exit(1)
    else:
        method = 'semantic'  # Default to semantic similarity
    
    auto_validate_results(method=method)
