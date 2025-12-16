from sentence_transformers import CrossEncoder

class Reranker:
    """
    Reranks retrieved chunks using a cross-encoder model
    for more accurate relevance scoring
    """
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the reranker
        
        Args:
            model_name: Cross-encoder model from sentence-transformers
        """
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✓ Reranker loaded")
    
    def rerank(self, query, documents, top_k=5):
        """
        Rerank documents based on query-document relevance
        
        Args:
            query: User's question
            documents: List of LangChain Document objects
            top_k: Number of top results to return
            
        Returns:
            List of reranked documents (top_k)
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score each pair
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        ranked = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top K documents
        return [doc for doc, score in ranked[:top_k]]
    
    def rerank_with_scores(self, query, documents, top_k=5):
        """
        Rerank and return both documents and their scores
        
        Returns:
            List of tuples (document, score)
        """
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        ranked = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked[:top_k]


# Example usage
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # Initialize reranker
    reranker = Reranker()
    
    # Example documents
    docs = [
        Document(page_content="The capital requirement is 10%", metadata={"source": "doc1"}),
        Document(page_content="Banks must report quarterly", metadata={"source": "doc2"}),
        Document(page_content="Capital adequacy ratio must be above 10%", metadata={"source": "doc3"})
    ]
    
    # Rerank
    query = "What is the capital requirement?"
    reranked = reranker.rerank(query, docs, top_k=2)
    
    print("\nReranked results:")
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. {doc.page_content[:50]}...")
