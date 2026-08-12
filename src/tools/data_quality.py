import pandas as pd


def analyze_data_quality(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Generate a basic data-quality report.
    """

    missing_values = (
        dataframe.isnull()
        .sum()
        .to_dict()
    )

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    numeric_columns = list(
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = list(
        dataframe.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns
    )

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "total_missing_values": sum(
            missing_values.values()
        ),
    }


def get_quality_status(
    quality_report: dict,
) -> str:
    """
    Return a simple overall data-quality status.
    """

    if (
        quality_report["total_missing_values"] == 0
        and quality_report["duplicate_rows"] == 0
    ):
        return "Good"

    if (
        quality_report["total_missing_values"] < 10
        and quality_report["duplicate_rows"] < 5
    ):
        return "Needs Attention"

    return "Poor"