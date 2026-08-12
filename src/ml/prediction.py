import pandas as pd
import numpy as np


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_prediction_features(
    dataframe: pd.DataFrame,
    target_column: str,
) -> list:
    """
    Return all feature columns used for prediction.
    """

    if dataframe is None:
        raise ValueError(
            "Dataset is required."
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    return [
        column
        for column in dataframe.columns
        if column != target_column
    ]


# ============================================================
# GET FEATURE TYPES
# ============================================================

def get_feature_metadata(
    dataframe: pd.DataFrame,
    target_column: str,
) -> list:
    """
    Generate metadata required to dynamically
    create prediction input fields.
    """

    features = get_prediction_features(
        dataframe,
        target_column,
    )

    metadata = []

    for column in features:

        series = dataframe[column]

        # ----------------------------------------------------
        # Numeric feature
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            metadata.append(
                {
                    "name": column,
                    "type": "numeric",
                    "dtype": str(
                        series.dtype
                    ),
                    "minimum": float(
                        series.min()
                    )
                    if not series.dropna().empty
                    else 0.0,
                    "maximum": float(
                        series.max()
                    )
                    if not series.dropna().empty
                    else 0.0,
                    "mean": float(
                        series.mean()
                    )
                    if not series.dropna().empty
                    else 0.0,
                }
            )

        # ----------------------------------------------------
        # Categorical feature
        # ----------------------------------------------------

        else:

            values = (
                series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            values = sorted(
                values
            )

            # Avoid an enormous dropdown.
            if len(values) > 100:

                values = values[:100]

            metadata.append(
                {
                    "name": column,
                    "type": "categorical",
                    "dtype": str(
                        series.dtype
                    ),
                    "values": values,
                }
            )

    return metadata


# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================

def build_prediction_dataframe(
    values: dict,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Convert user-entered prediction values into
    the dataframe format expected by sklearn.
    """

    row = {}

    for column in feature_columns:

        if column not in values:

            raise ValueError(
                f"Missing prediction value "
                f"for '{column}'."
            )

        row[column] = values[
            column
        ]

    return pd.DataFrame(
        [row],
        columns=feature_columns,
    )


# ============================================================
# MAKE PREDICTION
# ============================================================

def make_prediction(
    trained_pipeline,
    values: dict,
    feature_columns: list,
):
    """
    Generate a prediction using the trained
    sklearn pipeline.
    """

    if trained_pipeline is None:

        raise ValueError(
            "A trained model is required "
            "before making predictions."
        )

    prediction_dataframe = (
        build_prediction_dataframe(
            values,
            feature_columns,
        )
    )

    prediction = (
        trained_pipeline.predict(
            prediction_dataframe
        )
    )

    if len(prediction) == 0:

        raise ValueError(
            "The model returned no prediction."
        )

    result = prediction[0]

    # Convert numpy values to native Python.
    if isinstance(
        result,
        np.generic,
    ):

        result = result.item()

    return result