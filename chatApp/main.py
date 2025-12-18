"""
RAG Chat API - Complete Walkthrough
====================================
This file shows EXACTLY how:
1. We call the retrieval pipeline
2. We retrieve chunks from Milvus + BM25
3. Where those chunks go
4. How we send chunks + query to an LLM
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import os
import sys

# Add pipeline folder to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pdf-retrieval-pipeline'))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from hybrid_search import HybridSearcher

# For LLM - using Ollama (free, local) or OpenAI
# Uncomment ONE of these:
# from openai import OpenAI  # For OpenAI
import requests  # For Ollama (local LLM)


# ============================================================
# STEP 1: PYDANTIC SCHEMAS (What goes in/out)
# ============================================================

class ChatRequest(BaseModel):
    """What the user sends to us"""
    question: str                  # The user's question
    top_k: Optional[int] = 3       # How many chunks to retrieve


class ChunkInfo(BaseModel):
    """A single retrieved chunk"""
    content: str                   # The actual text
    source_file: Optional[str]     # Which PDF it came from
    page: Optional[int]            # Page number
    score: Optional[float]         # Relevance score


class ChatResponse(BaseModel):
    """What we send back"""
    question: str                  # Echo the question
    answer: str                    # LLM-generated answer
    chunks_used: List[ChunkInfo]   # The chunks we retrieved


# ============================================================
# STEP 2: THE RETRIEVER CLASS
# ============================================================
# This wraps your existing pipeline (Milvus + BM25)
# 
# HOW IT WORKS:
#   1. Load chunks.pkl (your pre-chunked documents)
#   2. Connect to Milvus (vector database with embeddings)
#   3. Build BM25 index (for keyword matching)
#   4. When you search: combine both methods for better results

class Retriever:
    """
    Wraps Milvus (semantic search) + BM25 (keyword search)
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("INITIALIZING RETRIEVER")
        print("="*60)
        
        # Path to your pipeline folder
        pipeline_dir = os.path.join(os.path.dirname(__file__), '..', 'pdf-retrieval-pipeline')
        
        # ---------------------------------------------------------
        # PART A: Load Milvus config (host, port, collection name)
        # ---------------------------------------------------------
        print("\n Loading Milvus config...")
        config_path = os.path.join(pipeline_dir, 'milvus_config.pkl')
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        print(f"   Collection: {self.config['collection_name']}")
        print(f"   Host: {self.config['host']}:{self.config['port']}")
        
        # ---------------------------------------------------------
        # PART B: Load the embedding model
        # ---------------------------------------------------------
        # This converts text → vectors (numbers)
        # Same model used when creating the index!
        print("\n Loading embedding model (BAAI/bge-large-en-v1.5)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        print(" Embedding model ready")
        
        # ---------------------------------------------------------
        # PART C: Connect to Milvus vector database
        # ---------------------------------------------------------
        # Milvus stores your document chunks as vectors
        print("\n🔌 Connecting to Milvus...")
        self.vector_db = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.config['collection_name'],
            connection_args={
                "host": self.config['host'],
                "port": self.config['port']
            }
        )
        print(" Connected to Milvus")
        
        # ---------------------------------------------------------
        # PART D: Initialize BM25 (keyword search)
        # ---------------------------------------------------------
        # BM25 finds documents with matching keywords
        # Combined with semantic search = better results
        print("\nInitializing BM25 (hybrid search)...")
        chunks_path = os.path.join(pipeline_dir, 'chunks.pkl')
        self.hybrid_searcher = HybridSearcher(chunks_file=chunks_path)
        
        print("\n" + "="*60)
        print("Retriever ready!")
        print("="*60 + "\n")
    
    def retrieve(self, question: str, top_k: int = 3) -> List[ChunkInfo]:
        """
        RETRIEVE CHUNKS FOR A QUESTION
        
        This is the main function that:
        1. Takes your question
        2. Searches Milvus (semantic) + BM25 (keywords)
        3. Returns the most relevant chunks
        
        WHERE DO CHUNKS COME FROM?
        - chunks.pkl = your PDFs split into small pieces
        - Milvus = stores these chunks as vectors for similarity search
        - BM25 = indexes these chunks for keyword matching
        """
        print(f"\nRetrieving chunks for: '{question}'")
        print(f"Requesting top {top_k} chunks...")
        
        # Call hybrid search (combines semantic + BM25)
        results = self.hybrid_searcher.search_with_scores(
            query=question,
            vector_db=self.vector_db,
            top_k=top_k,
            semantic_weight=0.7,  # 70% semantic (meaning)
            bm25_weight=0.3       # 30% keywords
        )
        
        # Convert to our ChunkInfo format
        chunks = []
        for i, (doc, score) in enumerate(results, 1):
            chunk = ChunkInfo(
                content=doc.page_content,
                source_file=doc.metadata.get('source_file'),
                page=doc.metadata.get('page'),
                score=round(score, 4)
            )
            chunks.append(chunk)
            
            # Print what we found
            print(f"\n   📄 Chunk {i} (score: {score:.4f}):")
            print(f"      Source: {chunk.source_file}")
            print(f"      Preview: {chunk.content[:100]}...")
        
        print(f"\n   ✅ Retrieved {len(chunks)} chunks")
        return chunks


# ============================================================
# STEP 3: THE LLM (Language Model)
# ============================================================
# This takes the chunks + question and generates an answer
#
# OPTIONS:
#   - Ollama (FREE, runs locally) ← We'll use this
#   - OpenAI (paid, cloud API)
#   - vLLM, HuggingFace, etc.

def generate_answer(question: str, chunks: List[ChunkInfo]) -> str:
    """
    GENERATE AN ANSWER USING AN LLM
    
    HOW IT WORKS:
    1. Take the retrieved chunks
    2. Build a prompt: "Here's context... Now answer this question..."
    3. Send to LLM
    4. Return the answer
    
    WHERE DO CHUNKS GO?
    - They become the "context" in the prompt
    - The LLM reads this context to answer your question
    """
    print("\n" + "="*60)
    print("GENERATING ANSWER WITH LLM")
    print("="*60)
    
    # ---------------------------------------------------------
    # PART A: Build the context from chunks
    # ---------------------------------------------------------
    # Combine all chunk content into one string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.source_file or "Unknown"
        context_parts.append(f"[Document {i} - {source}]\n{chunk.content}")
    
    context = "\n\n".join(context_parts)
    
    print(f"\nContext built from {len(chunks)} chunks")
    print(f"   Total context length: {len(context)} characters")
    
    # ---------------------------------------------------------
    # PART B: Build the prompt
    # ---------------------------------------------------------
    # This tells the LLM what to do
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    print(f"\nPrompt built ({len(prompt)} characters)")
    
    # ---------------------------------------------------------
    # PART C: Send to LLM
    # ---------------------------------------------------------
    # OPTION 1: Ollama (FREE, LOCAL)
    # Make sure Ollama is running: ollama serve
    # And you have a model: ollama pull llama2
    
    print("\nSending to LLM (Ollama)...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # or "mistral", "codellama", etc.
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            answer = response.json()["response"]
            print("Got response from LLM")
            return answer.strip()
        else:
            print(f"Ollama error: {response.status_code}")
            return f"LLM Error: {response.text}"
            
    except requests.exceptions.ConnectionError:
        print("   Ollama not running!")
        print("   Returning chunks as fallback...")
        
        # Fallback: just return the context if no LLM available
        fallback = "LLM not available. Here are the relevant chunks:\n\n"
        for i, chunk in enumerate(chunks, 1):
            fallback += f"**Chunk {i}** ({chunk.source_file}):\n{chunk.content[:500]}...\n\n"
        return fallback
    


# ============================================================
# STEP 4: THE API ENDPOINT
# ============================================================
# This orchestrates everything:
#   User Question → Retrieve Chunks → Send to LLM → Return Answer

app = FastAPI(
    title="RAG Chat API",
    description="Retrieval-Augmented Generation: Search docs + Generate answers",
    version="1.0.0"
)

# Enable CORS so the HTML UI can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever (initialized once when app starts)
retriever = None


@app.on_event("startup")
async def startup():
    """Initialize retriever when app starts"""
    global retriever
    retriever = Retriever()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    THE MAIN ENDPOINT - ORCHESTRATES EVERYTHING
    
    FLOW:
    1. User sends question
    2. We RETRIEVE relevant chunks from Milvus/BM25
    3. Chunks go INTO THE PROMPT sent to LLM
    4. LLM generates answer BASED ON chunks
    5. We return answer + chunks used
    """
    print("\n" + "🟢"*30)
    print(f"NEW REQUEST: {request.question}")
    print("🟢"*30)
    
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # STEP A: Retrieve chunks
    chunks = retriever.retrieve(
        question=request.question,
        top_k=request.top_k
    )
    
    # STEP B: Generate answer (chunks go to LLM here!)
    answer = generate_answer(
        question=request.question,
        chunks=chunks
    )
    
    # STEP C: Return everything
    return ChatResponse(
        question=request.question,
        answer=answer,
        chunks_used=chunks
    )


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "retriever_ready": retriever is not None}


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
