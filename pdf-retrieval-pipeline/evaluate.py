import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
import json
from reranker import Reranker

# EVALUATION QUESTIONS
EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "Under what conditions can a licensed bank grant accommodation to related parties using Letters of Credit (LCs) on behalf of Ministries or State-Owned Enterprises, and what additional reporting obligations apply?"
    },
    {
        "id": 2,
        "question": "What is the annual licence fee payable for a licensed commercial bank with total assets between Rs. 2,000 billion and Rs. 3,000 billion for the year 2026, and by when must it be paid?"
    },
    {
        "id": 3,
        "question": "List any four categories that are considered 'related parties' of a licensed bank under the Banking Act Determination No. 04 of 2024."
    },
    {
        "id": 4,
        "question": "What is the maximum percentage of accommodation a licensed bank may grant against quoted shares of public companies?"
    },
    {
        "id": 5,
        "question": "During the COVID-19 moratorium period, how is interest on deferred loan instalments treated, and what additional interest are banks allowed to charge on EMI loans?"
    },
    {
        "id": 6,
        "question": "What are the minimum capital surcharge requirements (HLA as a percentage of CET1) for Domestic Systemically Important Banks (D-SIBs) in Buckets 1, 2, and 3?"
    },
    {
        "id": 7,
        "question": "What are the Timelinesfor Reporting IT and Cybersecurity Incidents by licensed banks to the Central Bank of Sri Lanka as per the BSD Circular No. 2 of 2025?"
    },
    {
        "id": 8,
        "question": "By what date must licensed banks submit their Recovery Plans for 2025 to the Central Bank?"
    },
    {
        "id": 9,
        "question": "Which previous Banking Act Directions were revoked when the new Loan-to-Value Ratios for credit facilities granted in respect of motor vehicles came into effect in 2025?"
    },
    {
        "id": 10,
        "question": "What actions may the Central Bank impose if a Domestic Systemically Important Bank (D-SIB) fails to meet the minimum capital surcharge requirement within the specified timeframe?"
    },
    {
        "id": 11,
        "question": "What annual certification is required regarding Recovery Plans, who must provide it, and within what timeframe?"
    },
    {
        "id": 12,
        "question": "What is the maximum interest rate that licensed commercial banks and the National Savings Bank are allowed to offer or pay on foreign currency (FCY) deposits?"
    },
    {
        "id": 13,
        "question": "What criteria must Small and Medium Enterprises (SMEs) meet to qualify for the relief measures, and what are the key conditions attached to loan rescheduling?"
    },
    {
        "id": 14,
        "question": "What are the maximum limits applicable for accommodation granted to related parties of licensed banks, and how are these limits determined?"
    },
    {
        "id": 15,
        "question": "Within what timeframes must licensed banks publish quarterly financial information and annual audited financial statements?"
    },
    {
        "id": 16,
        "question": "What change was made to the maximum interest rate limits on foreign currency deposits of licensed commercial banks and the National Savings Bank in March 2022?"
    },
    {
        "id": 17,
        "question": "According to the May 2025 Monetary Policy Review, when is inflation expected to turn positive, and how are inflation expectations described?"
    },
    {
        "id": 18,
        "question": "What was the reported GDP growth rate for Q1-2025, and how did private sector credit perform according to the July 2025 review?"
    },
    {
        "id": 19,
        "question": "When did headline inflation turn positive after the deflationary period, and how long had deflation lasted prior to this change?"
    },
    {
        "id": 20,
        "question": "What concessions are licensed banks required to extend for lease facilities obtained by businesses and individuals in the passenger transportation sector affected by COVID-19?"
    }
]

# CONFIGURATION
TOP_K = 5  # Number of chunks to return (after reranking)
RETRIEVE_K = 15  # Number of chunks to retrieve before reranking
USE_RERANKING = True  # Set to False to disable reranking

def load_milvus_config():
    """Load Milvus connection info"""
    try:
        with open('milvus_config.pkl', 'rb') as f:
            config = pickle.load(f)
        return config
    except FileNotFoundError:
        print("❌ milvus_config.pkl not found. Run create_index.py first.")
        return None

def initialize_retriever():
    """Initialize Milvus connection and embeddings"""
    config = load_milvus_config()
    if not config:
        return None
    
    print("🔄 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"🔄 Connecting to Milvus at {config['host']}:{config['port']}...")
    connection_args = {
        "host": config['host'],
        "port": config['port']
    }
    
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=config['collection_name'],
        connection_args=connection_args
    )
    
    print("✅ Connected to Milvus\n")
    return vector_db

def retrieve_for_question(vector_db, question_data, reranker=None):
    """Retrieve top K chunks for a question"""
    print(f"\n{'='*80}")
    print(f"QUESTION {question_data['id']}")
    print(f"{'='*80}")
    print(f"{question_data['question']}\n")
    
    # Retrieve chunks
    if USE_RERANKING and reranker:
        # Retrieve more candidates for reranking
        print(f"🔍 Retrieving {RETRIEVE_K} candidates for reranking...")
        results = vector_db.similarity_search(
            question_data['question'],
            k=RETRIEVE_K
        )
        
        # Rerank to get top K
        print(f"🔄 Reranking to top {TOP_K}...")
        results = reranker.rerank(
            question_data['question'], 
            results, 
            top_k=TOP_K
        )
        print(f"✅ Reranking complete\n")
    else:
        # Direct retrieval without reranking
        results = vector_db.similarity_search(
            question_data['question'],
            k=TOP_K
        )
    
    print(f"📊 Final results: {len(results)} chunks:\n")
    
    # Store results for evaluation
    retrieved_chunks = []
    
    for i, doc in enumerate(results, 1):
        chunk_info = {
            "rank": i,
            "content": doc.page_content,
            "source_file": doc.metadata.get('source_file', 'N/A'),
            "relevant": None  # Will be marked manually
        }
        retrieved_chunks.append(chunk_info)
        
        # Display chunk
        print(f"\n{'─'*80}")
        print(f"RANK {i}")
        print(f"{'─'*80}")
        
        # Show source file
        print(f"📄 Source: {doc.metadata.get('source_file', 'N/A')}")
        
        # Show content
        print(f"\n📝 Content (first 500 chars):")
        print(f"{doc.page_content[:500]}...")
        print(f"\n💭 Full content length: {len(doc.page_content)} characters")
    
    return retrieved_chunks

def manual_evaluation():
    """Run evaluation and collect manual relevance judgments"""
    vector_db = initialize_retriever()
    if not vector_db:
        return
    
    # Initialize reranker if enabled
    reranker = None
    if USE_RERANKING:
        print("\n🔄 Initializing reranker...")
        reranker = Reranker()
        print("✅ Reranker ready\n")
    
    print(f"\n{'='*80}")
    print("RETRIEVAL EVALUATION")
    if USE_RERANKING:
        print(f"MODE: Two-stage retrieval (Retrieve {RETRIEVE_K} → Rerank to {TOP_K})")
    else:
        print(f"MODE: Direct retrieval (Top {TOP_K})")
    print(f"{'='*80}")
    print(f"📌 Evaluating {len(EVALUATION_QUESTIONS)} questions")
    print(f"📌 Retrieving top {TOP_K} chunks per question")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # Retrieve for each question
    for question_data in EVALUATION_QUESTIONS:
        retrieved_chunks = retrieve_for_question(vector_db, question_data, reranker)
        
        result = {
            "question_id": question_data['id'],
            "question": question_data['question'],
            "retrieved_chunks": retrieved_chunks
        }
        all_results.append(result)
    
    # Save results to JSON for manual annotation
    output_file = 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {output_file}")
    print(f"\n📝 Next steps:")
    print(f"  1. Review the results in {output_file}")
    print(f"  2. For each chunk, set 'relevant': true/false")
    print(f"  3. Run calculate_metrics.py to get evaluation scores")
    print(f"{'='*80}\n")

def quick_view():
    """Quick view of all questions and their top result"""
    vector_db = initialize_retriever()
    if not vector_db:
        return
    
    print(f"\n{'='*80}")
    print("QUICK EVALUATION VIEW (Top Result Only)")
    print(f"{'='*80}\n")
    
    for question_data in EVALUATION_QUESTIONS:
        print(f"\n📌 Q{question_data['id']}: {question_data['question'][:100]}...")
        
        results = vector_db.similarity_search(question_data['question'], k=1)
        if results:
            doc = results[0]
            print(f"   ✓ Top Result: {doc.metadata.get('source_file', 'N/A')}")
            print(f"   ✓ Type: {doc.metadata.get('content_metadata', {}).get('document_type', 'N/A')}")
            print(f"   ✓ Content: {doc.page_content[:150]}...\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_view()
    else:
        manual_evaluation()
