from pathlib import Path

import pandas as pd

from src.database.database import get_connection
from src.tools.data_quality import (
    analyze_data_quality,
)


# ==================================================
# SUPPORTED FILE TYPES
# ==================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}


# ==================================================
# CLEAN COLUMN NAMES
# ==================================================

def clean_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize column names for SQL and ML.

    Examples:

        ' Unit_Cost '
            ->
        'unit_cost'

        'Product Category'
            ->
        'product_category'

        'Product-Category'
            ->
        'product_category'
    """

    dataframe.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in dataframe.columns
    ]

    return dataframe


# ==================================================
# CLEAN NUMERIC / CURRENCY VALUES
# ==================================================

def clean_numeric_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert currency-formatted and numeric-looking
    values into proper numeric values.

    Examples:

        '$1,266.00 '  -> 1266.00
        '$420.00'     -> 420.00
        '1,250'       -> 1250.00
        '₹5,000'      -> 5000.00
        '($500.00)'   -> -500.00
    """

    cleaned = (
        series
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------
    # Handle empty values
    # ------------------------------------------------

    cleaned = cleaned.replace(
        {
            "": None,
            "nan": None,
            "NaN": None,
            "None": None,
            "null": None,
            "NULL": None,
            "N/A": None,
            "n/a": None,
        }
    )

    # ------------------------------------------------
    # Detect accounting-style negative values
    #
    # Example:
    # ($500.00) -> -500.00
    # ------------------------------------------------

    negative_mask = (
        cleaned.str.startswith("(")
        & cleaned.str.endswith(")")
    )

    # Remove parentheses
    cleaned = (
        cleaned
        .str.replace(
            "(",
            "",
            regex=False,
        )
        .str.replace(
            ")",
            "",
            regex=False,
        )
    )

    # ------------------------------------------------
    # Remove currency symbols
    # ------------------------------------------------

    cleaned = cleaned.str.replace(
        r"[$₹€£¥]",
        "",
        regex=True,
    )

    # ------------------------------------------------
    # Remove commas
    # ------------------------------------------------

    cleaned = cleaned.str.replace(
        ",",
        "",
        regex=False,
    )

    # ------------------------------------------------
    # Remove percentage symbols
    # ------------------------------------------------

    cleaned = cleaned.str.replace(
        "%",
        "",
        regex=False,
    )

    # ------------------------------------------------
    # Remove extra whitespace
    # ------------------------------------------------

    cleaned = cleaned.str.strip()

    # ------------------------------------------------
    # Convert to numeric
    # ------------------------------------------------

    numeric = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    # ------------------------------------------------
    # Restore negative accounting values
    # ------------------------------------------------

    numeric.loc[
        negative_mask & numeric.notna()
    ] *= -1

    return numeric


# ==================================================
# CONVERT NUMERIC COLUMNS
# ==================================================

def convert_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Automatically detect columns containing
    numeric or currency-formatted values.

    This prevents business metrics such as:

        Revenue
        Cost
        Profit
        Unit_Cost
        Unit_Price

    from being stored as TEXT in SQLite.
    """

    for column in dataframe.columns:

        # ------------------------------------------------
        # Already numeric
        # ------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            continue

        original = dataframe[column]

        # ------------------------------------------------
        # Attempt conversion
        # ------------------------------------------------

        converted = clean_numeric_series(
            original
        )

        # ------------------------------------------------
        # Count valid values
        # ------------------------------------------------

        non_null_original = (
            original.notna().sum()
        )

        non_null_converted = (
            converted.notna().sum()
        )

        # Nothing to evaluate
        if non_null_original == 0:
            continue

        # ------------------------------------------------
        # Conversion success ratio
        # ------------------------------------------------

        conversion_ratio = (
            non_null_converted
            / non_null_original
        )

        # ------------------------------------------------
        # Convert if at least 80% of the values
        # can be interpreted as numeric.
        # ------------------------------------------------

        if conversion_ratio >= 0.80:

            dataframe[column] = converted

    return dataframe


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset(
    file_path: str,
    table_name: str = "uploaded_data",
) -> dict:
    """
    Load CSV or Excel data into SQLite.

    Complete pipeline:

        File
          ↓
        Pandas
          ↓
        Remove empty rows
          ↓
        Clean column names
          ↓
        Clean numeric/currency values
          ↓
        Data quality analysis
          ↓
        SQLite
          ↓
        Return metadata
    """

    # ------------------------------------------------
    # Convert to Path
    # ------------------------------------------------

    path = Path(file_path)

    # ------------------------------------------------
    # Validate File
    # ------------------------------------------------

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # ------------------------------------------------
    # Detect File Type
    # ------------------------------------------------

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            "Unsupported file type. "
            "Please upload CSV or Excel files."
        )

    # ------------------------------------------------
    # Read Dataset
    # ------------------------------------------------

    if extension == ".csv":

        dataframe = pd.read_csv(
            path
        )

    elif extension in [".xlsx", ".xls"]:

        dataframe = pd.read_excel(
            path
        )

    # ------------------------------------------------
    # Remove Completely Empty Rows
    # ------------------------------------------------

    dataframe = dataframe.dropna(
        how="all"
    )

    # ------------------------------------------------
    # Validate Dataset
    # ------------------------------------------------

    if dataframe.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )

    # ------------------------------------------------
    # Clean Column Names
    # ------------------------------------------------

    dataframe = clean_column_names(
        dataframe
    )

    # ------------------------------------------------
    # Convert Numeric / Currency Columns
    # ------------------------------------------------

    dataframe = convert_numeric_columns(
        dataframe
    )

    # ------------------------------------------------
    # Data Quality Analysis
    # ------------------------------------------------

    quality_report = analyze_data_quality(
        dataframe
    )

    # ------------------------------------------------
    # Determine Numeric Columns
    # ------------------------------------------------

    numeric_columns = list(
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    # ------------------------------------------------
    # Determine Categorical Columns
    # ------------------------------------------------

    categorical_columns = list(
        dataframe.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns
    )

    # ------------------------------------------------
    # Store Dataset in SQLite
    # ------------------------------------------------

    connection = get_connection()

    try:

        dataframe.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )

    finally:

        connection.close()

    # ------------------------------------------------
    # Return Dataset Metadata
    # ------------------------------------------------

    return {
        "table_name": table_name,

        "rows": len(
            dataframe
        ),

        "columns": list(
            dataframe.columns
        ),

        "column_count": len(
            dataframe.columns
        ),

        "numeric_columns": numeric_columns,

        "categorical_columns": (
            categorical_columns
        ),

        "quality_report": quality_report,
    }