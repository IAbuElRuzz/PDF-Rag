# 📄 PDF RAG Tool with Ollama

This project is a **Retrieval-Augmented Generation (RAG)** tool that allows you to query your **PDF documents** using a **local Ollama LLM**.  
It extracts text from PDFs, embeds them with HuggingFace, stores them in a **FAISS vector database**, and answers your questions interactively.

You can choose between two PDF parsers:

- `main.py` → uses **PyPDF2**
- `run_pdf_plumber.py` → uses **pdfplumber** (often gives better extraction results)

---

## ✨ Features
- Extracts text from PDFs (`PyPDF2` or `pdfplumber`)
- Splits text into chunks with overlap for context-aware retrieval
- Stores embeddings in a **FAISS** index
- Uses HuggingFace `all-MiniLM-L6-v2` for embeddings
- Runs **local Ollama models** (e.g., `llama3`) for answering queries
- Interactive Q&A in the terminal
- Debug mode (`run_pdf_plumber.py`) shows retrieved context

---



## ✨ Features
- Extracts text from PDFs (`PyPDF2` or `pdfplumber`)
- Splits text into chunks with overlap for context-aware retrieval
- Stores embeddings in a **FAISS** index
- Uses HuggingFace `all-MiniLM-L6-v2` for embeddings
- Runs **local Ollama models** (e.g., `llama3`) for answering queries
- Interactive Q&A in the terminal
- Debug mode (`run_pdf_plumber.py`) shows retrieved context

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-rag-ollama.git
cd pdf-rag-ollama
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and run [Ollama](https://ollama.ai)
Make sure Ollama is installed and running. Pull a model (e.g., `llama3`):
```bash
ollama pull llama3
```

---

## 📂 Project Structure
```
pdf-rag-ollama/
│── main.py              # PyPDF2-based pipeline
│── run_pdf_plumber.py   # pdfplumber-based pipeline
│── requirements.txt     # Python dependencies
│── pdf_folder/          # Put your PDFs here
│── documents.pkl        # Cached documents (auto-generated)
│── faiss_index/         # FAISS index (auto-generated)
```

---

## ▶️ Usage

### 1. Add PDFs
Place your files inside the `pdf_folder/` directory.

### 2. Run the tool
**Using PyPDF2:**
```bash
python main.py
```

**Using pdfplumber (recommended):**
```bash
python run_pdf_plumber.py
```

### 3. Ask questions
Type your query in the terminal:
```
Your question: What are the key findings of the report?

Answer:
 The report highlights...
```

Type `exit` or `quit` to stop.

---

## 🛠 Debugging (Optional)
When using `run_pdf_plumber.py`, the script will also show the **top retrieved context** before answering:
```
🔎 Top matching context:

Match 1:
"The financial results for 2023 show a 12% increase..."

🤖 Answer:
The report concludes that revenue growth was driven by...
```

---

## ⚡ Requirements
- Python 3.9+
- [Ollama](https://ollama.ai) installed & running
- Dependencies:
  ```
  PyPDF2
  pdfplumber
  langchain
  langchain-community
  langchain-huggingface
  langchain-ollama
  faiss-cpu
  sentence-transformers
  ```

---

## 📌 Notes
- Cached files:  
  - `documents.pkl` → serialized documents  
  - `faiss_index/` → FAISS index for retrieval  
- If you add new PDFs, **delete both files** to force reindexing.
- Use `run_pdf_plumber.py` for better extraction on complex PDFs.

---

## 📜 License
MIT License. Free to use and modify.
