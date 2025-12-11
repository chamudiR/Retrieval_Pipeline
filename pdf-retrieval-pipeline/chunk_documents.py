import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LOAD DOCUMENTS FROM PREVIOUS STEP
def load_documents():
    """Load documents from step 1"""
    try:
        with open('documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        print(f"✓ Loaded {len(documents)} documents from step 1")
        return documents
    except FileNotFoundError:
        print("❌ documents.pkl not found. Run step1_load_documents.py first!")
        return []

# CHUNK MARKDOWN DOCUMENTS
def chunk_documents(documents):
    """Split markdown documents into chunks"""
    
    # Markdown-aware text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Adjust based on your needs
        chunk_overlap=200,      # Overlap to preserve context
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
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    
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
        print(f"\n{'='*70}")
        print(f"CHUNKS FROM: {source_file}")
        print(f"{'='*70}")
        print(f"Total chunks from this file: {len(filtered_chunks)}")
        display_chunks = filtered_chunks
    else:
        print(f"\n{'='*70}")
        print(f"ALL CHUNKS PREVIEW")
        print(f"{'='*70}")
        print(f"Total chunks: {len(chunks)}")
        display_chunks = chunks
    
    if not display_chunks:
        print("❌ No chunks to display")
        return
    
    print(f"\nShowing first {min(num_samples, len(display_chunks))} chunks:\n")
    
    for i, chunk in enumerate(display_chunks[:num_samples], 1):
        print(f"\n{'─'*70}")
        print(f"Chunk {i}")
        print(f"Source: {chunk.metadata.get('source_file', 'Unknown')}")
        print(f"Chunk Length: {len(chunk.page_content)} characters")
        print(f"{'─'*70}")
        print(chunk.page_content)
        print(f"{'─'*70}")


# ANALYZE CHUNK QUALITY
def analyze_chunks(chunks):
    """Analyze chunk statistics"""
    
    lengths = [len(c.page_content) for c in chunks]
    
    print(f"\n{'='*70}")
    print("CHUNK STATISTICS")
    print(f"{'='*70}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(lengths) / len(lengths):.0f} characters")
    print(f"Smallest chunk: {min(lengths)} characters")
    print(f"Largest chunk: {max(lengths)} characters")
    print(f"{'='*70}")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 2: CHUNK MARKDOWN DOCUMENTS")
    print("="*70)
    
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
    print(f"\n✓ Saved chunks to chunks.pkl")
    
    # Get unique source files
    source_files = list(set([c.metadata.get('source_file') for c in chunks]))
    
    print(f"\n📁 Available PDF files:")
    for i, file in enumerate(source_files, 1):
        num_chunks = len([c for c in chunks if c.metadata.get('source_file') == file])
        print(f"  {i}. {file} ({num_chunks} chunks)")
    
    # Interactive preview
    while True:
        print("\n" + "─"*70)
        choice = input("\nPreview options:\n  1. View chunks from specific PDF\n  2. View all chunks\n  3. View specific chunk number\n  4. Continue to next step\nChoice (1/2/3/4): ")
        
        if choice == '1':
            file_num = input(f"Enter PDF number (1-{len(source_files)}): ")
            try:
                file_idx = int(file_num) - 1
                if 0 <= file_idx < len(source_files):
                    num_samples = input("How many chunks to show? (default 3): ")
                    num = int(num_samples) if num_samples else 3
                    preview_chunks(chunks, source_files[file_idx], num_samples=num)
                else:
                    print("Invalid number")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == '2':
            num_samples = input("How many chunks to show? (default 3): ")
            num = int(num_samples) if num_samples else 3
            preview_chunks(chunks, num_samples=num)
        
        elif choice == '3':
            try:
                chunk_num = int(input(f"Enter chunk number (1-{len(chunks)}): "))
                if 1 <= chunk_num <= len(chunks):
                    preview_chunks([chunks[chunk_num-1]], num_samples=1)
                else:
                    print("Invalid chunk number")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice")
    
    print("\n✅ CHECKPOINT 2 COMPLETE")
    print("Run step3_create_index.py to continue")
