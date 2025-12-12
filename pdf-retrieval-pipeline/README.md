# PDF Retrieval Pipeline with Milvus

A complete PDF document retrieval system using Docling for OCR extraction, LangChain for text processing, and Milvus for vector search.

## Features

- 📄 **PDF Processing**: Extract text from PDFs (including scanned documents) using Docling with OCR
- 🧹 **Text Cleaning**: Automatic post-processing to fix common OCR errors (missing spaces, character misreadings)
- 📑 **Page-Level Tracking**: Preserves page numbers for accurate source attribution
- 🔍 **Semantic Search**: Vector-based search using Milvus and HuggingFace embeddings
- ⚡ **Optimized Chunking**: Markdown-aware text splitting for better context preservation

## Prerequisites

### Required Software

1. **Python 3.11+** (tested with Python 3.14)
2. **Docker Desktop** (for Milvus vector database)
3. **Tesseract OCR** (for document processing)
   - Windows: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Mac: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

### Verify Tesseract Installation

```bash
tesseract --version
```

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd pdf-retrieval-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

**Activate the environment:**

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Milvus (Vector Database)

From the project root directory (where `docker-compose.yml` is located):

```bash
cd ..
docker compose up -d
```

**Verify Milvus is running:**

```bash
docker compose ps
```

You should see three containers running:
- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`

## Project Structure

```
pdf-retrieval-pipeline/
├── documents/              # Place your PDF files here
├── load_documents.py       # Step 1: Extract text from PDFs
├── chunk_documents.py      # Step 2: Split into chunks
├── create_index.py         # Step 3: Create Milvus index
├── search.py              # Step 4: Search your documents
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### Step 1: Load and Extract PDFs

Place your PDF files in the `documents/` folder, then run:

```bash
python load_documents.py
```

This will:
- Extract text from PDFs using OCR
- Clean common OCR errors (missing spaces, character mistakes)
- Preserve page numbers
- Save to `documents.pkl`

### Step 2: Chunk Documents

```bash
python chunk_documents.py
```

This will:
- Split documents into semantic chunks
- Preserve metadata (source file, page numbers)
- Save to `chunks.pkl`

### Step 3: Create Vector Index

```bash
python create_index.py
```

This will:
- Generate embeddings using `BAAI/bge-base-en-v1.5`
- Create Milvus collection
- Index all chunks for fast retrieval

### Step 4: Search Your Documents

```bash
python search.py
```

Interactive search interface. Ask questions about your documents!

**Example queries:**
- "What are the key findings?"
- "Explain the methodology"
- "What are the conclusions?"

Results will show:
- Source file name
- Page number
- Relevant text content

## Configuration

### Chunk Size

Edit `chunk_documents.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust for your needs
    chunk_overlap=200,    # Context preservation
    ...
)
```

### Embedding Model

Edit `create_index.py` and `search.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"  # Change model here
)
```

Popular alternatives:
- `sentence-transformers/all-MiniLM-L6-v2` (faster, smaller)
- `sentence-transformers/all-mpnet-base-v2` (balanced)
- `BAAI/bge-large-en-v1.5` (more accurate, slower)

### OCR Settings

Edit `load_documents.py` to customize OCR behavior and text cleaning rules.

## Troubleshooting

### Tesseract Not Found

Make sure Tesseract is installed and in your PATH:

```bash
tesseract --version
```

### Milvus Connection Error

Check if Milvus is running:

```bash
docker compose ps
```

Restart if needed:

```bash
docker compose restart
```

### Out of Memory

- Reduce `chunk_size` in `chunk_documents.py`
- Process fewer documents at once
- Use a smaller embedding model

### Missing Page Numbers

Make sure you're using the updated `load_documents.py` that processes pages individually.

## Stopping Services

To stop Milvus:

```bash
docker compose down
```

To stop and remove all data:

```bash
docker compose down -v
```

## Features in Development

- [ ] Support for multiple languages
- [ ] Web interface
- [ ] Document update/deletion
- [ ] Advanced filtering
- [ ] Citation extraction

## License

MIT License

## Contributing

Pull requests welcome! Please open an issue first to discuss proposed changes.

## Support

For issues and questions, please open a GitHub issue.
