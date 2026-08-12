from src.database.database import get_connection


def get_table_schema(
    table_name: str = "uploaded_data",
) -> list[dict]:
    """
    Retrieve column information for a SQLite table.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            f"PRAGMA table_info({table_name})"
        )

        rows = cursor.fetchall()

    finally:

        connection.close()

    schema = []

    for row in rows:

        schema.append(
            {
                "column_name": row[1],
                "data_type": row[2],
                "nullable": not bool(row[3]),
                "primary_key": bool(row[5]),
            }
        )

    return schema


def format_schema_for_llm(
    table_name: str = "uploaded_data",
) -> str:
    """
    Convert the database schema into a format
    that can be provided to the LLM.
    """

    schema = get_table_schema(
        table_name
    )

    if not schema:
        return (
            f"No schema found for table "
            f"'{table_name}'."
        )

    lines = [
        f"Table: {table_name}",
        "",
        "Columns:",
    ]

    for column in schema:

        lines.append(
            f"- {column['column_name']} "
            f"({column['data_type']})"
        )

    return "\n".join(lines)