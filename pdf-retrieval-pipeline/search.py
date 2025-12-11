import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus


# LOAD CONFIGURATION
def load_config():
    """Load Milvus configuration"""
    try:
        with open('milvus_config.pkl', 'rb') as f:
            config = pickle.load(f)
        return config
    except FileNotFoundError:
        print("milvus_config.pkl not found.")
        return None


# CONNECT TO VECTOR STORE
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
    
    print(" Connected!\n")
    return vector_db


# SEARCH FUNCTION
def search(vector_db, query, k=3):
    """Search for relevant chunks"""
    
    print(f"🔍 Query: {query}")
    
    results = vector_db.similarity_search(query, k=k)
    
    if not results:
        print("No results found")
        return
    
    for i, doc in enumerate(results, 1):
        print(f"📄 Result {i}/{len(results)}")
        print(f"Source File: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"\n📝 Content:")
        print(doc.page_content)


# MAIN
if __name__ == "__main__":
    # Load config
    config = load_config()
    if not config:
        exit()
    
    # Connect to vector store
    try:
        vector_db = connect_to_vector_store(config)
    except Exception as e:
        print(f"Error {e}")
        exit()
    
    # Interactive search
    print("💡 Tips:")
    print("  - Question?")
    print("  - Type 'quit' or 'exit' to stop")
    
    while True:
        try:
            query = input("🔍 Enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                num_results = input("Number of results (default 3): ").strip()
                k = int(num_results) if num_results else 3
                search(vector_db, query, k=k)
            
        except KeyboardInterrupt:
            break
        except ValueError:
            print("Invalid number, using default (3)")
            search(vector_db, query, k=3)
        except Exception as e:
            print(f" Error: {e}")
