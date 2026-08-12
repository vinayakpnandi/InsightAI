from pathlib import Path

from src.database.loader import load_dataset
from src.database.schema import (
    format_schema_for_llm,
)


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("INSIGHTAI DATASET TEST")
    print("=" * 60)

    print()
    print(
        "Put a CSV or XLSX file inside:"
    )

    print(
        "data/raw/"
    )

    print()

    file_name = input(
        "Enter dataset filename: "
    ).strip()

    file_path = Path(
        "data/raw"
    ) / file_name

    # ----------------------------------------------
    # Load Dataset
    # ----------------------------------------------

    result = load_dataset(
        file_path=str(file_path),
        table_name="uploaded_data",
    )

    print()
    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Table: {result['table_name']}"
    )

    print(
        f"Rows: {result['rows']}"
    )

    print(
        f"Columns: {result['column_count']}"
    )

    print(
        f"Column names: {result['columns']}"
    )

    # ----------------------------------------------
    # Display Schema
    # ----------------------------------------------

    print()
    print("=" * 60)
    print("DATABASE SCHEMA")
    print("=" * 60)

    print()

    print(
        format_schema_for_llm(
            "uploaded_data"
        )
    )