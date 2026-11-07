# PDF-Rag

**Ask questions over local PDF collections with semantic retrieval and Ollama.**

PDF-Rag extracts PDF text, splits it into overlapping chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and retrieves context from an in-memory FAISS index before generating an answer with a local `llama3` model.

This is a terminal prototype. It does not provide OCR, evaluated answer accuracy, a service API, or production deployment guarantees.

## Quick start

Use **Python 3.11 or 3.12** and an installed, running [Ollama](https://ollama.com/download) service. Run all commands from the repository root.

```sh
git clone https://github.com/IAbuElRuzz/PDF-Rag.git
cd PDF-Rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
ollama pull llama3
python run.py
# Optional: select a directory, model, or retrieval count
python run.py --pdf-dir ./pdf_folder --model llama3 --top-k 4
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. If Ollama is not running, start its application or run `ollama serve` in a separate terminal.

Place text-based PDFs in `pdf_folder/` before starting. The repository includes a sample PDF. Type a question at the terminal prompt; use `exit` or `quit` to stop.

The initial setup requires network access to install packages and download Ollama and embedding models. Once downloaded, inference uses the local Ollama service; model installation is not an offline operation.

### Choose an extraction pipeline

| Entry point | Extraction | Additional behavior |
| --- | --- | --- |
| `python run.py` | PyPDF2 | Terminal question answering |
| `python run_pdf_plumber.py` | pdfplumber | Displays two semantic-search matches before answering |

The second script's displayed matches come from a separate search; they are not guaranteed to be exactly the documents selected by RetrievalQA. Both scripts print the answer’s retrieved source filenames and page numbers.

## Retrieval architecture

```mermaid
flowchart LR
    A[PDF folder] --> B[Extract text]
    B --> C[Overlapping chunks]
    C --> D[MiniLM embeddings]
    D --> E[FAISS index]
    Q[Question] --> R[Retrieve context]
    E --> R
    R --> L[Ollama llama3]
    L --> O[Terminal answer]
```

Both scripts accept `--pdf-dir`, `--model`, and `--top-k`. They use 1,000-character chunks with 200-character overlap, Hugging Face embeddings, an in-memory FAISS index, and LangChain RetrievalQA. Page numbers and filenames are attached as chunk metadata and shown with each answer. The LangChain dependencies are constrained to the 0.3 family to retain the legacy imports used by this prototype; migration to current APIs is separate work. The embedding dependencies also retain a compatible Transformers 4.x and NumPy 1.x stack for older CPU wheels. These are compatibility ranges, not a fully locked dependency graph; this legacy stack still needs ongoing dependency and security maintenance.

## Index and troubleshooting

The current scripts rebuild the FAISS index in memory at startup. This avoids loading pickle-based document caches or FAISS indexes that require unsafe deserialization, and ensures changed PDFs are reflected on the next run. The tradeoff is that PDF extraction and embedding repeat each time. A future persistent cache should use a safe, versioned format and verify its source files and embedding configuration.

- **`ModuleNotFoundError`:** activate the virtual environment and install the complete requirements file. Both PDF parsers are included.
- **`langchain.chains` import errors:** use the declared 0.3 compatibility ranges in a fresh Python 3.11/3.12 environment; unconstrained LangChain upgrades can remove legacy APIs.
- **Connection refused or missing model:** confirm Ollama is running and `ollama list` includes `llama3`.
- **No extracted text:** scanned/image-only PDFs require OCR, which is not implemented.
- **Poor answers:** inspect extracted text and retrieved context. An answer is model output, not verified evidence.

The tracked FAISS binary alone is not a complete reusable cache. Building from the PDFs is the supported first-run workflow.

## Development checks

```sh
python -m pip check
python -m compileall -q run.py run_pdf_plumber.py pdf_rag.py
python run.py --help
python run_pdf_plumber.py --help
python -c "import sentence_transformers; import pdf_rag, run, run_pdf_plumber; assert pdf_rag.load_documents"
```

The import check does not download model weights or call Ollama. An end-to-end question-answering demonstration requires local models and is not established by this check.

## License status

The original README stated MIT, but this repository currently has no standalone LICENSE file. An explicit license file and copyright statement should be added by the owner to clarify reuse terms.
