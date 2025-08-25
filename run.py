import os
import pickle
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


def load_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Reading {filename} ...")
            reader = PdfReader(os.path.join(folder_path, filename))
            text = "".join(page.extract_text() or "" for page in reader.pages)
            texts.append(text)
    return texts


def split_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents(texts)


def save_state(documents, db, docs_path, index_path):
    print("Saving state...")
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
    db.save_local(index_path)
    print("State saved.")


def main():
    folder_path = "./pdf_folder"  # Change this to your PDF folder path
    docs_path = "documents.pkl"
    index_path = "faiss_index"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load or build documents + FAISS index
    if os.path.exists(docs_path) and os.path.exists(index_path):
        print("Loading cached documents and FAISS index...")
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
        db = FAISS.load_local(index_path, embeddings)
    else:
        print("Processing PDFs and creating index...")
        texts = load_pdfs(folder_path)
        documents = split_texts(texts)
        db = FAISS.from_documents(documents, embeddings)
        save_state(documents, db, docs_path, index_path)

    print("Initializing local LLM (Ollama)...")
    llm = OllamaLLM(model="llama3")  # Use your desired Ollama model

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    print("Ready! Ask your questions (type 'exit' to quit)")

    try:
        while True:
            query = input("Your question: ")
            if query.lower() in ["exit", "quit"]:
                break
            outputs = qa.invoke(query)
            print("\nAnswer:\n", outputs["result"])
            # Optional: print sources
            # print("\nSources:\n", outputs["source_documents"])

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Save state before exiting on crash
        save_state(documents, db, docs_path, index_path)
        print("Exiting due to error.")


if __name__ == "__main__":
    main()
