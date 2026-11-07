"""Shared PDF ingestion and local retrieval helpers."""
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(folder: Path, parser: str):
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder does not exist: {folder}")
    documents = []
    for path in sorted(folder.glob("*.pdf")):
        print(f"Reading {path.name} ...")
        if parser == "plumber":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [(page.extract_text() or "") for page in pdf.pages]
        else:
            from PyPDF2 import PdfReader
            pages = [(page.extract_text() or "") for page in PdfReader(path).pages]
        for page_number, text in enumerate(pages, 1):
            if text.strip():
                documents.append({"text": text, "source": path.name, "page": page_number})
    if not documents:
        raise ValueError(f"No extractable text found in PDFs under {folder}; scanned PDFs need OCR, which is not implemented.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents(
        [item["text"] for item in documents],
        metadatas=[{"source": item["source"], "page": item["page"]} for item in documents],
    )


def build_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return FAISS.from_documents(documents, embeddings)


def print_sources(documents):
    seen = set()
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            print(f"- {source}, page {page}")
            seen.add(key)
