import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


def build_preprocessor(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Build a preprocessing pipeline for numeric
    and categorical features.
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    X = dataframe.drop(
        columns=[target_column]
    )

    numeric_features = list(
        X.select_dtypes(
            include="number"
        ).columns
    )

    categorical_features = list(
        X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor