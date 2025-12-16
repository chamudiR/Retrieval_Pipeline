from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, results, top_k=5):
    # Score each result
    pairs = [[query, doc.page_content] for doc in results]
    scores = model.predict(pairs)
    
    # Rerank by score
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]