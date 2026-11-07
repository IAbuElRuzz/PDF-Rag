"""pdfplumber variant that previews retrieved context."""
import argparse
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

from pdf_rag import MODEL_NAME, build_index, load_documents, print_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdf_folder"), help="Directory containing PDFs")
    parser.add_argument("--model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--top-k", type=int, default=4, help="Retrieved chunks per question (default: 4)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        documents = load_documents(args.pdf_dir, "plumber")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Embedding {len(documents)} text chunks with {MODEL_NAME} (first run may download the model)...")
    database = build_index(documents)
    qa = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model=args.model),
        retriever=database.as_retriever(search_kwargs={"k": args.top_k}),
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
        matches = database.similarity_search(query, k=min(args.top_k, 2))
        print("\nRetrieved context:")
        for match in matches:
            print(f"\n[{match.metadata['source']} p.{match.metadata['page']}]\n{match.page_content[:500]}")
        try:
            result = qa.invoke({"query": query})
        except Exception as error:
            print(f"Query failed: {error}")
            continue
        print("\nAnswer:\n" + result["result"])
        print("\nSources:")
        print_sources(result.get("source_documents", []))
        print()


if __name__ == "__main__":
    main()
