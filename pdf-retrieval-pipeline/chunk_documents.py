import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LOAD DOCUMENTS FROM PREVIOUS STEP
def load_documents():
    """Load documents from step 1"""
    try:
        with open('documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        print(f" Loaded {len(documents)} documents from step 1")
        return documents
    except FileNotFoundError:
        print(" documents.pkl not found.")
        return []

# CHUNK MARKDOWN DOCUMENTS
def chunk_documents(documents):
    """Split markdown documents into chunks"""
    
    # Markdown-aware text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,       # Adjust based on your needs
        chunk_overlap=500,      # Overlap to preserve context
        separators=[
            "\n## ",    # Split on H2 headers
            "\n### ",   # Split on H3 headers
            "\n\n",     # Split on paragraph breaks
            "\n",       # Split on line breaks
            ". ",       # Split on sentences
            " ",        # Split on words
            ""
        ],
        length_function=len,
        is_separator_regex=False
    )
    
    print("\nChunking markdown documents...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Show chunk distribution
    print(f"\n{'='*70}")
    print("CHUNK DISTRIBUTION BY SOURCE")
    print(f"{'='*70}")
    
    source_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get('source_file', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in source_counts.items():
        print(f"  {source}: {count} chunks")
    
    return chunks


# PREVIEW CHUNKS
def preview_chunks(chunks, source_file=None, num_samples=3):
    """Preview chunks, optionally filtered by source file"""
    
    # Filter by source if specified
    if source_file:
        filtered_chunks = [c for c in chunks if c.metadata.get('source_file') == source_file]
        print(f"CHUNKS FROM: {source_file}")
        
        print(f"Total chunks from this file: {len(filtered_chunks)}")
        display_chunks = filtered_chunks
    else:
        print(f"ALL CHUNKS PREVIEW")
        print(f"Total chunks: {len(chunks)}")
        display_chunks = chunks
    
    if not display_chunks:
        print(" No chunks to display")
        return
    
    print(f"\nShowing first {min(num_samples, len(display_chunks))} chunks:\n")
    
    for i, chunk in enumerate(display_chunks[:num_samples], 1):
        print(f"\nChunk {i}")
        print(f"Chunk {i}")
        print(f"Source: {chunk.metadata.get('source_file', 'Unknown')}")
        print(f"Chunk Length: {len(chunk.page_content)} characters")
        print(chunk.page_content)


# ANALYZE CHUNK QUALITY
def analyze_chunks(chunks):
    """Analyze chunk statistics"""
    
    lengths = [len(c.page_content) for c in chunks]
    
    print("CHUNK STATISTICS")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(lengths) / len(lengths):.0f} characters")
    print(f"Smallest chunk: {min(lengths)} characters")
    print(f"Largest chunk: {max(lengths)} characters")


# MAIN  
if __name__ == "__main__":
    print("STEP 2: CHUNK MARKDOWN DOCUMENTS")
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        exit()
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Analyze chunks
    analyze_chunks(chunks)
    
    # Save chunks for next step
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    print(f"\n Saved chunks to chunks.pkl")
    
    # Get unique source files
    source_files = list(set([c.metadata.get('source_file') for c in chunks]))
    
    print(f"\n Available PDF files:")
    for i, file in enumerate(source_files, 1):
        num_chunks = len([c for c in chunks if c.metadata.get('source_file') == file])
        print(f"  {i}. {file} ({num_chunks} chunks)")
    