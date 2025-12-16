import os
import re
import logging
import json
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from pypdf import PdfReader
from datetime import datetime

# Enable detailed logging to see what Docling uses
logging.basicConfig(level=logging.INFO)

# CONFIGURATION
PDF_FOLDER = "./documents"
METADATA_OUTPUT = "./metadata_extracted.json"

# REGEX RULES FOR METADATA EXTRACTION
REGEX_RULES = {
    'title': [
        r'^#\s+(.+)$',  # Markdown heading
        r'\*\*(.{10,100}?)\*\*',  # Bold text (10-100 chars)
        r'^(.+)$'  # First non-empty line
    ],
    'document_type': {
        'circular': r'(?i)\bcircular\b',
        'directions': r'(?i)\bdirection[s]?\b',
        'determination': r'(?i)\bdetermination\b',
        'act': r'(?i)\bact\b',
        'report': r'(?i)\breport\b',
        'press_release': r'(?i)\bpress\s+release\b',
        'gazette': r'(?i)\bgazette\b'
    },
    'document_number': [
        r'(?i)(?:no\.?|number)\s*[:\s]*(\d+)',  # "No. 1" or "Number: 1"
        r'(?i)circular\s+no\.?\s*(\d+)',  # "Circular No. 1"
        r'(?i)directions?\s+no\.?\s*(\d+)'  # "Directions No. 1"
    ],
    'year': r'\b(202[0-9])\b',  # 2020-2029
    'publication_date': [
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4})',  # 02 December2025 or 02 December 2025
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',  # December 02, 2025
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # 02/12/2025
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})'  # 2025-12-02
    ]
}

NO_OF_YEAR_RE = re.compile(r'(?i)\bno\.?\s*0*(\d+)\s*of\s*(\d{4})\b')

def extract_metadata_from_content(markdown_content, filename, num_pages):
    """Extract metadata from markdown content using regex rules (header-first precedence)."""

    # Prefer line-based header to avoid body noise
    lines_all = [line.strip() for line in markdown_content.split('\n') if line.strip()]

    # Header: first ~80 non-empty lines (good for Docling output)
    header_lines = lines_all[:80]
    header_text_lines = "\n".join(header_lines)

    # Also keep your original 5000-char window as a fallback/extra context
    header_text = markdown_content[:5000]

    # --------------------
    # Document type (header-first)
    # --------------------
    doc_type = 'N/A'
    for dtype, pattern in REGEX_RULES['document_type'].items():
        if re.search(pattern, header_text_lines) or re.search(pattern, header_text):
            doc_type = dtype
            break

    # --------------------
    # Document number + year (strong rule first)
    # --------------------
    doc_number = 'N/A'
    doc_year = 'N/A'

    strong = NO_OF_YEAR_RE.search(header_text_lines) or NO_OF_YEAR_RE.search(header_text)
    if strong:
        doc_number = strong.group(1)          # e.g., "02"
        doc_year = int(strong.group(2))       # e.g., 2025
    else:
        # Fallback to your existing document number rules
        for pattern in REGEX_RULES['document_number']:
            match = re.search(pattern, header_text_lines) or re.search(pattern, header_text)
            if match:
                doc_number = match.group(1)
                break

        # Fallback year rule ONLY if strong rule didn't find year
        year_match = re.search(REGEX_RULES['year'], header_text_lines) or re.search(REGEX_RULES['year'], header_text)
        if year_match:
            doc_year = int(year_match.group(1))

    # --------------------
    # Publication date (header-first)
    # --------------------
    publication_date = 'N/A'
    for pattern in REGEX_RULES['publication_date']:
        match = re.search(pattern, header_text_lines, re.IGNORECASE)
        if not match:
            match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            publication_date = match.group(1).strip()
            publication_date = re.sub(r'\s+', ' ', publication_date)  # normalize OCR spacing
            break

    # --------------------
    # Subject (first substantial paragraph, same as yours but slightly safer)
    # --------------------
    subject = 'N/A'
    for line in lines_all[:30]:  # widened from 20 → 30 (optional)
        if line.startswith('#') or line.startswith('*'):
            continue
        if len(line) > 50:
            subject = line[:200]
            break

    # --------------------
    # Language (same logic as yours)
    # --------------------
    language = 'en' if ('_e' in filename.lower() or 'english' in filename.lower()) else 'N/A'

    # --------------------
    # Build structured metadata (same shape as yours)
    # --------------------
    metadata = {
        'filename': filename,
        'file_metadata': {
            'subject': subject,
            'page_count': num_pages,
            'language': language
        },
        'content_metadata': {
            'document_type': doc_type,
            'document_number': doc_number,
            'document_year': doc_year,
            'publication_date': publication_date
        }
    }

    return metadata

# TEXT CLEANING FUNCTION
def clean_ocr_text(text):
    """Fix common OCR errors like missing spaces"""
    
    text = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\(|\[)', r'\1 \2', text)
    text = re.sub(r'(\)|\])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text.strip()

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
    
    # Show what Docling will use
    print("\n📋 Docling Configuration:")
    print("  - Default settings (no OCR optimization)")
    print("  - OCR: Auto-detect (uses available engine)")
    print("  - Table extraction: Enabled")
    print("  - Layout analysis: Enabled")
    
    all_documents = []
    all_metadata = []  # Store all metadata for JSON export
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"\n Processing: {pdf_file}")
        
        try:
            # Convert PDF to structured document
            result = converter.convert(pdf_path)
            
            # Export to Markdown
            markdown_content = result.document.export_to_markdown()
            
            # Clean OCR errors (fix missing spaces)
            cleaned_content = clean_ocr_text(markdown_content)
            
            # Get page count
            num_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 0
            
            # Extract metadata from content using regex
            print(f"  📋 Extracting metadata using regex rules...")
            pdf_metadata = extract_metadata_from_content(cleaned_content, pdf_file, num_pages)
            
            # Show key metadata
            print(f"    Type: {pdf_metadata['content_metadata']['document_type']}")
            print(f"    Number: {pdf_metadata['content_metadata']['document_number']}")
            print(f"    Year: {pdf_metadata['content_metadata']['document_year']}")
            print(f"    Date: {pdf_metadata['content_metadata']['publication_date']}")
            
            # Store metadata for JSON export
            all_metadata.append(pdf_metadata)
            
            print(f"  ✓ Converted successfully")
            print(f"  ✓ Extracted {len(markdown_content)} characters")
            print(f"  ✓ Cleaned text (fixed spacing issues)")
            
            # Create a LangChain Document with the cleaned markdown and metadata
            doc = Document(
                page_content=cleaned_content,
                metadata={
                    'source_file': pdf_file,
                    'format': 'markdown',
                    'num_pages': len(result.document.pages) if hasattr(result.document, 'pages') else 'unknown',
                    'cleaned': True,
                    # Include PDF metadata
                    **pdf_metadata
                }
            )
            
            all_documents.append(doc)
            
            print(f" Document added to collection")
            
        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    
    # Save metadata to JSON file
    print(f"\n💾 Saving metadata to {METADATA_OUTPUT}...")
    with open(METADATA_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata saved successfully ({len(all_metadata)} documents)")
    
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