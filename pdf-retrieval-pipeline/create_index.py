import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus


# CONFIGURATION
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "pdf_knowledge_base"

# LOAD CHUNKS
def load_chunks():
    """Load chunks from step 2"""
    try:
        with open('chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from step 2")
        return chunks
    except FileNotFoundError:
        print("chunks.pkl not found.")
        return []


# CREATE VECTOR STORE
def create_vector_store(chunks):
    """Create embeddings and index in Milvus"""
    
    # Initialize embedding model
    print("\nLoading embedding model...")
    print("(This may take 1-2 minutes on first run)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",  # Much better results
        model_kwargs={'device': 'cpu'}
    )
    
    print("Embedding model loaded")
    
    # Test embedding
    print("\nTesting embedding model...")
    test_vector = embeddings.embed_query("test")
    print(f"Embedding dimension: {len(test_vector)}")
    
    # Connect to Milvus
    print(f"\nConnecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    
    connection_args = {
        "host": MILVUS_HOST,
        "port": MILVUS_PORT
    }
    
    # Create vector store
    print(f"\nCreating collection '{COLLECTION_NAME}'...")
    
    try:
        vector_db = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args=connection_args,
            drop_old=True
        )
        
        print("\n✅ Indexing complete!")
        
        # Verify indexing
        print("\nVerifying index...")
        test_results = vector_db.similarity_search("test query", k=1)
        print(f"Index working! Retrieved {len(test_results)} result(s)")
        
        return vector_db, embeddings
        
    except Exception as e:
        print(f"\nError creating index: {e}")
        return None, None


# SAVE CONNECTION INFO
def save_connection_info():
    """Save connection info for search script"""
    config = {
        'host': MILVUS_HOST,
        'port': MILVUS_PORT,
        'collection_name': COLLECTION_NAME
    }
    with open('milvus_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("Saved connection info to milvus_config.pkl")


# MAIN
if __name__ == "__main__": 
    # Load chunks
    chunks = load_chunks()
    
    if not chunks:
        exit()
    
    # Create vector store
    vector_db, embeddings = create_vector_store(chunks)
    
    if vector_db:
        # Save connection info
        save_connection_info()
    else:
        print("\nFaileed")
