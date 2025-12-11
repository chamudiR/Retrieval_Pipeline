import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        print(f"✓ Loaded {len(chunks)} chunks from step 2")
        return chunks
    except FileNotFoundError:
        print("❌ chunks.pkl not found. Run step2_chunk_documents.py first!")
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
    
    print("✓ Embedding model loaded")
    
    # Test embedding
    print("\nTesting embedding model...")
    test_vector = embeddings.embed_query("test")
    print(f"✓ Embedding dimension: {len(test_vector)}")
    
    # Connect to Milvus
    print(f"\nConnecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    
    connection_args = {
        "host": MILVUS_HOST,
        "port": MILVUS_PORT
    }
    
    # Create vector store
    print(f"\nCreating collection '{COLLECTION_NAME}'...")
    print(f"Indexing {len(chunks)} chunks (this may take several minutes)...")
    
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
        print(f"✓ Index working! Retrieved {len(test_results)} result(s)")
        
        return vector_db, embeddings
        
    except Exception as e:
        print(f"\n❌ Error creating index: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Milvus is running: docker compose ps")
        print("  2. Check Milvus logs: docker logs milvus-standalone")
        return None, None

# ============================================
# SAVE CONNECTION INFO
# ============================================
def save_connection_info():
    """Save connection info for search script"""
    config = {
        'host': MILVUS_HOST,
        'port': MILVUS_PORT,
        'collection_name': COLLECTION_NAME
    }
    with open('milvus_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("✓ Saved connection info to milvus_config.pkl")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 3: CREATE EMBEDDINGS & INDEX IN MILVUS")
    print("="*70)
    
    # Load chunks
    chunks = load_chunks()
    
    if not chunks:
        exit()
    
    # Create vector store
    vector_db, embeddings = create_vector_store(chunks)
    
    if vector_db:
        # Save connection info
        save_connection_info()
        
        print("\n" + "="*70)
        print("✅ CHECKPOINT 3 COMPLETE")
        print("="*70)
        print("\nYour PDF knowledge base is ready!")
        print("Run step4_search.py to start searching")
    else:
        print("\n❌ Failed to create index. Please check errors above.")
