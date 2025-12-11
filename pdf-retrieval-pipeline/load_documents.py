import os
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

# ============================================
# CONFIGURATION
# ============================================
PDF_FOLDER = "./documents"

# ============================================
# LOAD PDFs WITH DOCLING
# ============================================
def load_pdfs_with_docling():
    """Load and convert PDFs using Docling"""
    
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder not found: {PDF_FOLDER}")
        return []
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {PDF_FOLDER}")
        return []
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    
    # Initialize Docling converter
    print("\nInitializing Docling converter...")
    converter = DocumentConverter()
    
    all_documents = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"\n{'='*70}")
        print(f"📄 Processing: {pdf_file}")
        print(f"{'='*70}")
        
        try:
            # Convert PDF to structured document
            print("  Converting with Docling (this may take a minute)...")
            result = converter.convert(pdf_path)
            
            # Export to Markdown
            markdown_content = result.document.export_to_markdown()
            
            print(f"  ✓ Converted successfully")
            print(f"  ✓ Extracted {len(markdown_content)} characters")
            
            # Create a LangChain Document with the full markdown
            doc = Document(
                page_content=markdown_content,
                metadata={
                    'source_file': pdf_file,
                    'format': 'markdown',
                    'num_pages': len(result.document.pages) if hasattr(result.document, 'pages') else 'unknown'
                }
            )
            
            all_documents.append(doc)
            
            print(f"  ✓ Document added to collection")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✓ Total documents loaded: {len(all_documents)}")
    print(f"{'='*70}")
    
    return all_documents

# ============================================
# PREVIEW FUNCTION
# ============================================
def preview_documents(documents, num_chars=1000):
    """Preview extracted markdown from documents"""
    
    print(f"\n{'='*70}")
    print(f"DOCUMENT PREVIEW (MARKDOWN FORMAT)")
    print(f"{'='*70}")
    
    if not documents:
        print("❌ No documents to preview")
        return
    
    for i, doc in enumerate(documents, 1):
        print(f"\n{'─'*70}")
        print(f"📄 Document {i}/{len(documents)}")
        print(f"{'─'*70}")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Pages: {doc.metadata.get('num_pages', 'Unknown')}")
        print(f"Total Length: {len(doc.page_content)} characters")
        print(f"\n📝 Content Preview (first {num_chars} characters):")
        print(f"{'─'*70}")
        print(doc.page_content[:num_chars])
        print(f"\n... (showing {num_chars}/{len(doc.page_content)} characters)")
        print(f"{'─'*70}")

# ============================================
# SAVE MARKDOWN TO FILE (OPTIONAL)
# ============================================
def save_markdown_files(documents):
    """Save extracted markdown to separate files for inspection"""
    
    output_folder = "extracted_markdown"
    os.makedirs(output_folder, exist_ok=True)
    
    for doc in documents:
        source_file = doc.metadata.get('source_file', 'unknown.pdf')
        md_filename = source_file.replace('.pdf', '.md')
        md_path = os.path.join(output_folder, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(doc.page_content)
        
        print(f"  ✓ Saved: {md_path}")
    
    print(f"\n✓ All markdown files saved to '{output_folder}' folder")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 1: LOAD & CONVERT PDFs WITH DOCLING")
    print("="*70)
    print("\n💡 Docling will:")
    print("   - Extract text from scanned PDFs (OCR)")
    print("   - Preserve tables, headings, and structure")
    print("   - Convert everything to clean Markdown")
    print("="*70)
    
    # Load documents
    documents = load_pdfs_with_docling()
    
    if documents:
        # Save to pickle for next step
        import pickle
        with open('documents.pkl', 'wb') as f:
            pickle.dump(documents, f)
        print(f"\n✓ Saved {len(documents)} documents to documents.pkl")
        
        # Preview
        preview_documents(documents, num_chars=1500)
        
        # Ask if user wants to save markdown files
        while True:
            response = input("\n\nSave markdown files for inspection? (y/n): ").lower()
            if response == 'y':
                save_markdown_files(documents)
                break
            elif response == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
        
        # Interactive preview
        while True:
            response = input("\n\nView full content of a specific document? (y/n): ").lower()
            if response == 'y':
                try:
                    doc_num = int(input(f"Enter document number (1-{len(documents)}): "))
                    if 1 <= doc_num <= len(documents):
                        print(f"\n{'='*70}")
                        print(f"FULL CONTENT - Document {doc_num}")
                        print(f"{'='*70}\n")
                        print(documents[doc_num-1].page_content)
                        print(f"\n{'='*70}")
                    else:
                        print("Invalid document number")
                except ValueError:
                    print("Please enter a valid number")
            else:
                break
        
        print("\n✅ CHECKPOINT 1 COMPLETE")
        print("Run step2_chunk_documents.py to continue")
    else:
        print("\n❌ No documents loaded. Check your PDF folder and try again.")
