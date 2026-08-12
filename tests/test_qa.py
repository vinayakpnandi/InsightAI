from src.rag.qa import answer_question


if __name__ == "__main__":

    question = input(
        "Ask a question about your document: "
    )

    result = answer_question(question)

    print()
    print("=" * 60)
    print("INSIGHTAI ANSWER")
    print("=" * 60)

    print()
    print(result["answer"])

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:
        print(
            f"Source: {source['source']} | "
            f"Page: {source['page']}"
        )