import numpy as np
import pandas as pd


# ============================================================
# GET TRANSFORMED FEATURE NAMES
# ============================================================

def _get_feature_names(
    preprocessor,
    number_of_features: int,
) -> np.ndarray:
    """
    Get feature names after sklearn preprocessing.
    """

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        feature_names = np.array(
            [
                f"feature_{i}"
                for i in range(
                    number_of_features
                )
            ]
        )

    return np.asarray(
        feature_names,
        dtype=str,
    )


# ============================================================
# CLEAN FEATURE NAMES
# ============================================================

def _clean_feature_names(
    feature_names,
) -> list:

    cleaned = []

    for name in feature_names:

        name = str(name)

        name = name.replace(
            "num__",
            "",
        )

        name = name.replace(
            "cat__",
            "",
        )

        cleaned.append(name)

    return cleaned


# ============================================================
# GET MODEL IMPORTANCE
# ============================================================

def get_feature_importance(
    pipeline,
    dataframe: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Extract feature importance from a trained sklearn Pipeline.

    Supports:

    1. Tree-based models:
       feature_importances_

    2. Linear models:
       coef_

    Returns:

        feature
        importance
        importance_percent
        method
    """

    if pipeline is None:

        raise ValueError(
            "Trained model pipeline is required."
        )

    if dataframe is None:

        raise ValueError(
            "Dataframe is required."
        )

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    # ========================================================
    # GET PREPROCESSOR
    # ========================================================

    if hasattr(
        pipeline,
        "named_steps",
    ):

        preprocessor = (
            pipeline
            .named_steps
            .get("preprocessor")
        )

        model = (
            pipeline
            .named_steps
            .get("model")
        )

    else:

        preprocessor = None
        model = pipeline

    if model is None:

        raise ValueError(
            "Could not find trained model "
            "inside the pipeline."
        )

    # ========================================================
    # TREE-BASED MODEL
    # ========================================================

    if hasattr(
        model,
        "feature_importances_",
    ):

        importances = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

        method = (
            "Model Feature Importance"
        )

    # ========================================================
    # LINEAR MODEL
    # ========================================================

    elif hasattr(
        model,
        "coef_",
    ):

        coefficients = np.asarray(
            model.coef_,
            dtype=float,
        )

        # --------------------------------------------
        # Regression
        # --------------------------------------------

        if coefficients.ndim == 1:

            importances = np.abs(
                coefficients
            )

        # --------------------------------------------
        # Multiclass classification
        # --------------------------------------------

        else:

            importances = np.mean(
                np.abs(
                    coefficients
                ),
                axis=0,
            )

        method = (
            "Absolute Model Coefficients"
        )

    else:

        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_percent",
                "method",
            ]
        )

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    if preprocessor is not None:

        feature_names = _get_feature_names(
            preprocessor,
            len(importances),
        )

    else:

        feature_names = np.array(
            [
                f"feature_{i}"
                for i in range(
                    len(importances)
                )
            ]
        )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(feature_names) != len(
        importances
    ):

        feature_names = np.array(
            [
                f"feature_{i}"
                for i in range(
                    len(importances)
                )
            ]
        )

    # ========================================================
    # CLEAN NAMES
    # ========================================================

    feature_names = (
        _clean_feature_names(
            feature_names
        )
    )

    # ========================================================
    # CREATE RESULT
    # ========================================================

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "method": method,
        }
    )

    # ========================================================
    # REMOVE INVALID VALUES
    # ========================================================

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    result = result.dropna(
        subset=[
            "importance"
        ]
    )

    # ========================================================
    # CALCULATE PERCENTAGE
    # ========================================================

    total_importance = (
        result[
            "importance"
        ].sum()
    )

    if total_importance > 0:

        result[
            "importance_percent"
        ] = (
            result["importance"]
            / total_importance
            * 100
        )

    else:

        result[
            "importance_percent"
        ] = 0.0

    # ========================================================
    # SORT
    # ========================================================

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# TOP FEATURES
# ============================================================

def get_top_features(
    feature_importance: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top N features.
    """

    if feature_importance is None:

        return pd.DataFrame()

    if feature_importance.empty:

        return feature_importance.copy()

    return (
        feature_importance
        .head(n)
        .copy()
    )