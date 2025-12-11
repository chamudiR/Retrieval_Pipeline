import os
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

# CONFIGURATION
PDF_FOLDER = "./documents"

# LOAD PDFs WITH DOCLING
def load_pdfs_with_docling():
    """Load and convert PDFs using Docling"""
    
    if not os.path.exists(PDF_FOLDER):
        print(f"Folder not found: {PDF_FOLDER}")
        return []
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
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
        print(f"\n Processing: {pdf_file}")
        
        try:
            # Convert PDF to structured document
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
            
            print(f" Document added to collection")
            
        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Total documents loaded: {len(all_documents)}")
    
    return all_documents

# PREVIEW FUNCTION
def preview_documents(documents, num_chars=1000):
    """Preview extracted markdown from documents"""
    
    print(f"DOCUMENT PREVIEW (MARKDOWN FORMAT)")
    
    if not documents:
        print("No documents to preview")
        return
    
    for i, doc in enumerate(documents, 1):
        print(f" Document {i}/{len(documents)}")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Pages: {doc.metadata.get('num_pages', 'Unknown')}")
        print(f"Total Length: {len(doc.page_content)} characters")
        print(f"\n Content Preview (first {num_chars} characters):")
        print(doc.page_content[:num_chars])
        print(f"\n... (showing {num_chars}/{len(doc.page_content)} characters)")
        


# SAVE MARKDOWN TO FILE (OPTIONAL)
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
        
        print(f" Saved: {md_path}")
    
    print(f"\nAll markdown files saved to '{output_folder}' folder")


# MAIN

if __name__ == "__main__":

    # Load documents
    documents = load_pdfs_with_docling()
    
    if documents:
        # Save to pickle for next step
        import pickle
        with open('documents.pkl', 'wb') as f:
            pickle.dump(documents, f)
        print(f"\nSaved {len(documents)} documents to documents.pkl")
        
        # Preview
        preview_documents(documents, num_chars=1500)
        save_markdown_files(documents)
        
        print("Done")
    else:
        print("\n No documents loaded.")
