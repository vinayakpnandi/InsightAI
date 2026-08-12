import re

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from src.ml.models import (
    get_classification_models,
    get_regression_models,
)

from src.ml.preprocessor import (
    build_preprocessor,
)


# ============================================================
# CURRENCY / NUMERIC TARGET DETECTION
# ============================================================

def _convert_numeric_like_target(
    target: pd.Series,
) -> tuple[pd.Series, bool]:
    """
    Detect object/string targets that actually contain
    numeric or currency values.

    Examples:

        "$2,320.00" -> 2320.00
        "$1,043.00" -> 1043.00
        "₹1,25,000" -> 125000.00
        "1,250.50"  -> 1250.50

    Returns:

        converted_target
        is_numeric_like
    """

    # Already numeric
    if pd.api.types.is_numeric_dtype(target):

        return (
            target.copy(),
            True,
        )

    # Boolean should remain classification
    if pd.api.types.is_bool_dtype(target):

        return (
            target.copy(),
            False,
        )

    # Work with strings
    cleaned = (
        target
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove common currency symbols / separators
    # --------------------------------------------------------

    normalized = (
        cleaned
        .str.replace(
            r"[\$,₹,€£¥]",
            "",
            regex=True,
        )
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.strip()
    )

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    numeric = pd.to_numeric(
        normalized,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Calculate conversion ratio
    # --------------------------------------------------------

    valid_mask = target.notna()

    valid_count = int(
        valid_mask.sum()
    )

    if valid_count == 0:

        return (
            target.copy(),
            False,
        )

    converted_count = int(
        numeric[valid_mask].notna().sum()
    )

    conversion_ratio = (
        converted_count
        / valid_count
    )

    # --------------------------------------------------------
    # Detect numeric-like target
    #
    # We require at least 90% of non-null values to
    # successfully convert.
    # --------------------------------------------------------

    if conversion_ratio >= 0.90:

        return (
            numeric.astype(float),
            True,
        )

    return (
        target.copy(),
        False,
    )


# ============================================================
# DETECT ML PROBLEM TYPE
# ============================================================

def detect_problem_type(
    dataframe: pd.DataFrame,
    target_column: str,
) -> str:
    """
    Determine whether the target is a classification
    or regression problem.

    Rules:

    1. Numeric target -> regression
    2. Currency-like numeric string target -> regression
    3. Text/categorical target -> classification
    4. Boolean target -> classification
    """

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )

    target = dataframe[
        target_column
    ]

    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    if pd.api.types.is_bool_dtype(
        target
    ):

        return "classification"

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        target
    ):

        return "regression"

    # --------------------------------------------------------
    # Currency / numeric-like string
    # --------------------------------------------------------

    _, is_numeric_like = (
        _convert_numeric_like_target(
            target
        )
    )

    if is_numeric_like:

        return "regression"

    # --------------------------------------------------------
    # Categorical / text
    # --------------------------------------------------------

    if (
        target.dtype == "object"
        or str(target.dtype)
        == "category"
    ):

        return "classification"

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return "classification"


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with a currency-like
    target converted to numeric values.

    This ensures the actual ML model receives:

        "$2,320.00"

    as:

        2320.0
    """

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )

    dataframe = dataframe.copy()

    target = dataframe[
        target_column
    ]

    converted_target, is_numeric_like = (
        _convert_numeric_like_target(
            target
        )
    )

    if is_numeric_like:

        dataframe[
            target_column
        ] = converted_target

    return dataframe


# ============================================================
# TRAIN ML MODELS
# ============================================================

def train_model(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Train multiple candidate ML models and select
    the best-performing model.

    Supports:

        Classification
        Regression

    Currency-like targets are automatically converted
    into numeric values and treated as regression.
    """

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if dataframe is None:

        raise ValueError(
            "No dataset was provided."
        )

    if dataframe.empty:

        raise ValueError(
            "The dataset is empty."
        )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    dataframe = prepare_target(
        dataframe,
        target_column,
    )

    # --------------------------------------------------------
    # Minimum dataset size
    # --------------------------------------------------------

    if len(dataframe) < 20:

        raise ValueError(
            "The dataset contains fewer than 20 rows. "
            "At least 20 rows are recommended for "
            "ML experimentation."
        )

    # --------------------------------------------------------
    # Remove missing target rows
    # --------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=[
            target_column
        ]
    )

    if dataframe.empty:

        raise ValueError(
            "No valid rows remain after removing "
            "missing target values."
        )

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = dataframe.drop(
        columns=[
            target_column
        ]
    )

    y = dataframe[
        target_column
    ]

    # --------------------------------------------------------
    # Validate target variability
    # --------------------------------------------------------

    if y.nunique() < 2:

        raise ValueError(
            f"The target column '{target_column}' "
            "contains fewer than two unique values. "
            "A model cannot be trained."
        )

    # --------------------------------------------------------
    # Detect problem type
    # --------------------------------------------------------

    problem_type = detect_problem_type(
        dataframe=dataframe,
        target_column=target_column,
    )

    print()
    print(
        "[InsightAI] Target:",
        target_column,
    )

    print(
        "[InsightAI] Problem type:",
        problem_type,
    )

    if problem_type == "regression":

        print(
            "[InsightAI] Currency/numeric target "
            "detected. Using regression models."
        )

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        dataframe=dataframe,
        target_column=target_column,
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    stratify = None

    if problem_type == "classification":

        class_counts = y.value_counts()

        if (
            len(class_counts) > 1
            and class_counts.min() >= 2
        ):

            stratify = y

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    )

    # --------------------------------------------------------
    # Candidate models
    # --------------------------------------------------------

    if problem_type == "classification":

        models = (
            get_classification_models()
        )

    else:

        models = (
            get_regression_models()
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    best_model = None
    best_score = None
    best_name = None

    # --------------------------------------------------------
    # Train candidates
    # --------------------------------------------------------

    for name, model in models.items():

        try:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            predictions = (
                pipeline.predict(
                    X_test
                )
            )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            if (
                problem_type
                == "classification"
            ):

                score = (
                    accuracy_score(
                        y_test,
                        predictions,
                    )
                )

                metric_name = (
                    "accuracy"
                )

                model_result = {
                    "model": name,
                    "metric": metric_name,
                    "score": float(
                        score
                    ),
                    "accuracy": float(
                        score
                    ),
                }

            # ------------------------------------------------
            # Regression
            # ------------------------------------------------

            else:

                r2 = r2_score(
                    y_test,
                    predictions,
                )

                mae = (
                    mean_absolute_error(
                        y_test,
                        predictions,
                    )
                )

                mse = (
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )

                rmse = (
                    mse ** 0.5
                )

                score = r2

                metric_name = "r2"

                model_result = {
                    "model": name,
                    "metric": metric_name,
                    "score": float(
                        score
                    ),
                    "r2": float(
                        r2
                    ),
                    "mae": float(
                        mae
                    ),
                    "rmse": float(
                        rmse
                    ),
                }

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                model_result
            )

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_model = pipeline
                best_name = name

        except Exception as error:

            results.append(
                {
                    "model": name,
                    "metric": "error",
                    "score": None,
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if best_model is None:

        raise ValueError(
            "All candidate ML models failed to train. "
            "Please check the dataset and target column."
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "problem_type": problem_type,

        "target_column": target_column,

        "best_model": best_model,

        "best_model_name": best_name,

        "best_score": float(
            best_score
        ),

        "results": results,
    }