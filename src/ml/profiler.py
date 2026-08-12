import pandas as pd


def profile_dataset(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Generate a basic profile of an uploaded dataset.
    """

    if dataframe.empty:
        raise ValueError(
            "Dataset is empty."
        )

    profile = {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "column_names": list(
            dataframe.columns
        ),
        "numeric_columns": list(
            dataframe.select_dtypes(
                include="number"
            ).columns
        ),
        "categorical_columns": list(
            dataframe.select_dtypes(
                include=["object", "category", "bool"]
            ).columns
        ),
        "missing_values": (
            dataframe.isnull()
            .sum()
            .to_dict()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
    }

    return profile