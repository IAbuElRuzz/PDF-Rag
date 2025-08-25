import os
import pickle
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama  # fallback if langchain_ollama fails


def load_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Reading {filename} ...")
            with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
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
    folder_path = "./pdf_folder"
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
    llm = Ollama(model="llama3")

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

            # DEBUG: Show top retrieved context
            print("\n🔎 Top matching context:")
            matches = db.similarity_search(query, k=2)
            for i, match in enumerate(matches, 1):
                print(f"\nMatch {i}:\n{match.page_content[:500]}...\n")

            # Invoke LLM
            outputs = qa.invoke(query)
            print("\n🤖 Answer:\n", outputs["result"])

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        save_state(documents, db, docs_path, index_path)
        print("State saved before exit.")


if __name__ == "__main__":
    main()
