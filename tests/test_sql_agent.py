from src.database.sql_agent import ask_sql_agent


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("INSIGHTAI SQL AGENT")
    print("=" * 60)

    question = input(
        "\nAsk a question about your dataset: "
    )

    try:

        result = ask_sql_agent(
            question=question,
            table_name="uploaded_data",
        )

        print()
        print("=" * 60)
        print("GENERATED SQL")
        print("=" * 60)

        print(result["sql"])

        print()
        print("=" * 60)
        print("QUERY RESULT")
        print("=" * 60)

        print(
            result["columns"]
        )

        for row in result["rows"]:
            print(row)

        print()
        print("=" * 60)
        print("INSIGHTAI ANSWER")
        print("=" * 60)

        print()
        print(result["answer"])

    except Exception as error:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(error)