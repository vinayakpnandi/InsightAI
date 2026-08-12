from src.rag.retriever import get_retriever


if __name__ == "__main__":

    question = input(
        "Ask a question about your document: "
    )

    retriever = get_retriever(k=4)

    documents = retriever.invoke(question)

    print()
    print("=" * 60)
    print("RETRIEVED DOCUMENT CHUNKS")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):

        print(f"\n--- Chunk {i} ---")

        print(
            f"Source: "
            f"{document.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page', 'Unknown')}"
        )

        print()

        print(document.page_content)