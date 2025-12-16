"""
Hybrid Search: Combines dense (semantic) and sparse (BM25) retrieval
"""
import pickle
from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.documents import Document

class HybridSearcher:
    """
    Combines BM25 (keyword matching) with semantic search (embeddings)
    for better retrieval accuracy
    """
    
    def __init__(self, chunks_file='chunks.pkl'):
        """
        Initialize hybrid searcher with chunks
        
        Args:
            chunks_file: Path to pickle file containing chunks
        """
        print("🔄 Loading chunks for BM25 indexing...")
        with open(chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"📚 Loaded {len(self.chunks)} chunks")
        
        # Create BM25 index
        print("🔄 Building BM25 index...")
        self._build_bm25_index()
        print("✅ BM25 index ready\n")
    
    def _build_bm25_index(self):
        """Build BM25 index from chunks"""
        # Tokenize documents (simple word splitting)
        tokenized_corpus = [
            doc.page_content.lower().split() 
            for doc in self.chunks
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def hybrid_search(self, query: str, vector_db, top_k=5, 
                     semantic_weight=0.7, bm25_weight=0.3):
        """
        Perform hybrid search combining semantic and BM25
        
        Args:
            query: Search query
            vector_db: Milvus vector database instance
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            
        Returns:
            List of top K documents
        """
        # Get more candidates than needed
        retrieve_k = top_k * 3
        
        # 1. Get semantic search results
        semantic_results = vector_db.similarity_search_with_score(
            query, 
            k=retrieve_k
        )
        
        # 2. Get BM25 scores for all chunks
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 3. Normalize scores to 0-1 range
        # Normalize semantic scores (lower distance = higher similarity)
        semantic_dict = {}
        if semantic_results:
            max_distance = max(score for _, score in semantic_results)
            min_distance = min(score for _, score in semantic_results)
            distance_range = max_distance - min_distance if max_distance > min_distance else 1
            
            for doc, distance in semantic_results:
                # Convert distance to similarity (invert and normalize)
                similarity = 1 - ((distance - min_distance) / distance_range)
                doc_content = doc.page_content
                semantic_dict[doc_content] = similarity
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized_bm25 = {
            self.chunks[i].page_content: score / max_bm25 
            for i, score in enumerate(bm25_scores)
        }
        
        # 4. Combine scores
        combined_scores = {}
        
        # Get all unique documents from both methods
        all_docs = set(semantic_dict.keys()) | set(normalized_bm25.keys())
        
        for doc_content in all_docs:
            semantic_score = semantic_dict.get(doc_content, 0)
            bm25_score = normalized_bm25.get(doc_content, 0)
            
            # Weighted combination
            combined_scores[doc_content] = (
                semantic_weight * semantic_score + 
                bm25_weight * bm25_score
            )
        
        # 5. Sort by combined score and get top K
        sorted_docs = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # 6. Return Document objects
        result_docs = []
        for doc_content, score in sorted_docs:
            # Find the original document
            for chunk in self.chunks:
                if chunk.page_content == doc_content:
                    result_docs.append(chunk)
                    break
        
        return result_docs
    
    def search_with_scores(self, query: str, vector_db, top_k=5,
                          semantic_weight=0.7, bm25_weight=0.3):
        """
        Hybrid search that returns documents with their scores
        
        Returns:
            List of tuples (document, combined_score)
        """
        retrieve_k = top_k * 3
        
        # Get semantic results
        semantic_results = vector_db.similarity_search_with_score(query, k=retrieve_k)
        
        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores
        semantic_dict = {}
        if semantic_results:
            max_distance = max(score for _, score in semantic_results)
            min_distance = min(score for _, score in semantic_results)
            distance_range = max_distance - min_distance if max_distance > min_distance else 1
            
            for doc, distance in semantic_results:
                similarity = 1 - ((distance - min_distance) / distance_range)
                semantic_dict[doc.page_content] = similarity
        
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized_bm25 = {
            self.chunks[i].page_content: score / max_bm25 
            for i, score in enumerate(bm25_scores)
        }
        
        # Combine scores
        combined_scores = {}
        all_docs = set(semantic_dict.keys()) | set(normalized_bm25.keys())
        
        for doc_content in all_docs:
            semantic_score = semantic_dict.get(doc_content, 0)
            bm25_score = normalized_bm25.get(doc_content, 0)
            combined_scores[doc_content] = (
                semantic_weight * semantic_score + 
                bm25_weight * bm25_score
            )
        
        # Sort and return
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        result = []
        for doc_content, score in sorted_docs:
            for chunk in self.chunks:
                if chunk.page_content == doc_content:
                    result.append((chunk, score))
                    break
        
        return result


# Example usage
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Milvus
    
    # Load config
    with open('milvus_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # Connect to Milvus
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=config['collection_name'],
        connection_args={"host": config['host'], "port": config['port']}
    )
    
    # Initialize hybrid searcher
    hybrid = HybridSearcher()
    
    # Test search
    query = "What is the capital requirement?"
    results = hybrid.hybrid_search(query, vector_db, top_k=5)
    
    print(f"\nHybrid search results for: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
