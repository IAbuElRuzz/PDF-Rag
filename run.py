"""Question-answering over PDFs with local Ollama inference."""
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

from pdf_rag import build_index, load_documents, print_sources


def main():
    documents = load_documents(Path("pdf_folder"), "pypdf2")
    print(f"Embedding {len(documents)} text chunks (first run may download the model)...")
    database = build_index(documents)
    qa = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="llama3"),
        retriever=database.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    print("Ready. Ask a question; type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = qa.invoke({"query": query})
        print("\nAnswer:\n" + result["result"])
        print("\nSources:")
        print_sources(result.get("source_documents", []))
        print()


if __name__ == "__main__":
    main()
