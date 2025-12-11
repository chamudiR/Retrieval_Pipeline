import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# ============================================
# LOAD CONFIGURATION
# ============================================
def load_config():
    """Load Milvus configuration"""
    try:
        with open('milvus_config.pkl', 'rb') as f:
            config = pickle.load(f)
        return config
    except FileNotFoundError:
        print("❌ milvus_config.pkl not found. Run step3_create_index.py first!")
        return None

# ============================================
# CONNECT TO VECTOR STORE
# ============================================
def connect_to_vector_store(config):
    """Connect to existing Milvus collection"""
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",  # Much better results
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Connecting to Milvus collection '{config['collection_name']}'...")
    
    connection_args = {
        "host": config['host'],
        "port": config['port']
    }
    
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=config['collection_name'],
        connection_args=connection_args
    )
    
    print("✓ Connected!\n")
    return vector_db

# ============================================
# SEARCH FUNCTION
# ============================================
def search(vector_db, query, k=3):
    """Search for relevant chunks"""
    
    print(f"\n{'='*70}")
    print(f"🔍 Query: {query}")
    print(f"{'='*70}")
    
    results = vector_db.similarity_search(query, k=k)
    
    if not results:
        print("No results found")
        return
    
    for i, doc in enumerate(results, 1):
        print(f"\n{'─'*70}")
        print(f"📄 Result {i}/{len(results)}")
        print(f"{'─'*70}")
        print(f"Source File: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"\n📝 Content:")
        print(f"{'─'*70}")
        print(doc.page_content)
        print(f"{'─'*70}")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 4: SEARCH YOUR PDF KNOWLEDGE BASE")
    print("="*70 + "\n")
    
    # Load config
    config = load_config()
    if not config:
        exit()
    
    # Connect to vector store
    try:
        vector_db = connect_to_vector_store(config)
    except Exception as e:
        print(f"❌ Error connecting to Milvus: {e}")
        print("\nMake sure Milvus is running: docker compose ps")
        exit()
    
    # Interactive search
    print("💡 Tips:")
    print("  - Ask questions about your PDF content")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Press Ctrl+C to force quit\n")
    
    while True:
        try:
            query = input("🔍 Enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query:
                num_results = input("Number of results (default 3): ").strip()
                k = int(num_results) if num_results else 3
                search(vector_db, query, k=k)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except ValueError:
            print("Invalid number, using default (3)")
            search(vector_db, query, k=3)
        except Exception as e:
            print(f"❌ Error: {e}")
