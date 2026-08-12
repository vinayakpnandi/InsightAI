import pandas as pd

from src.ml.profiler import (
    profile_dataset,
)

from src.ml.predictor import (
    detect_problem_type,
    prepare_target,
    train_model,
)

from src.ml.leakage import (
    analyze_target_leakage,
)

from src.ml.explainability import (
    get_feature_importance,
    get_top_features,
)


# ============================================================
# DATASET PROFILING
# ============================================================

def get_dataset_profile(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Generate a complete dataset profile.
    """

    if dataframe is None:

        raise ValueError(
            "No dataset was provided."
        )

    if dataframe.empty:

        raise ValueError(
            "The dataset is empty."
        )

    return profile_dataset(
        dataframe
    )


# ============================================================
# TARGET INFORMATION
# ============================================================

def get_target_information(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Analyze the selected target column.

    Currency-like targets are converted to numeric
    for ML analysis.
    """

    if dataframe is None:

        raise ValueError(
            "No dataset was provided."
        )

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    prepared_dataframe = prepare_target(
        dataframe,
        target_column,
    )

    target = prepared_dataframe[
        target_column
    ]

    # --------------------------------------------------------
    # Detect problem
    # --------------------------------------------------------

    problem_type = detect_problem_type(
        dataframe=prepared_dataframe,
        target_column=target_column,
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    information = {

        "target_column": target_column,

        "data_type": str(
            target.dtype
        ),

        "problem_type": problem_type,

        "unique_values": int(
            target.nunique()
        ),

        "missing_values": int(
            target.isnull().sum()
        ),
    }

    # ========================================================
    # REGRESSION
    # ========================================================

    if (
        problem_type
        == "regression"
    ):

        information.update(
            {
                "minimum": float(
                    target.min()
                ),

                "maximum": float(
                    target.max()
                ),

                "mean": float(
                    target.mean()
                ),

                "median": float(
                    target.median()
                ),
            }
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    else:

        value_counts = (
            target
            .value_counts()
            .head(10)
            .to_dict()
        )

        information[
            "class_distribution"
        ] = {
            str(key): int(value)
            for key, value
            in value_counts.items()
        }

    return information


# ============================================================
# COMPLETE ML ANALYSIS
# ============================================================

def run_ml_analysis(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Run the complete InsightAI ML pipeline.

    Pipeline:

        Dataset
           ↓
        Profile
           ↓
        Target analysis
           ↓
        Leakage detection
           ↓
        Model training
           ↓
        Feature importance
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if dataframe is None:

        raise ValueError(
            "No dataset was provided."
        )

    if dataframe.empty:

        raise ValueError(
            "The dataset is empty."
        )

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    # ========================================================
    # PROFILE
    # ========================================================

    profile = get_dataset_profile(
        dataframe
    )

    # ========================================================
    # TARGET ANALYSIS
    # ========================================================

    target_information = (
        get_target_information(
            dataframe,
            target_column,
        )
    )

    # ========================================================
    # LEAKAGE ANALYSIS
    # ========================================================

    leakage = analyze_target_leakage(
        dataframe,
        target_column,
    )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    training_result = train_model(
        dataframe=dataframe,
        target_column=target_column,
    )

    # ========================================================
    # BEST TRAINED PIPELINE
    # ========================================================

    trained_model = (
        training_result[
            "best_model"
        ]
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        get_feature_importance(
            pipeline=trained_model,
            dataframe=dataframe,
            target_column=target_column,
        )
    )

    # ========================================================
    # TOP FEATURES
    # ========================================================

    top_features = (
        get_top_features(
            feature_importance,
            n=10,
        )
    )

    # ========================================================
    # RETURN COMPLETE RESULT
    # ========================================================

    return {

        # Dataset
        "profile": profile,

        # Target
        "target": target_information,

        # Leakage
        "leakage": leakage,

        # ML
        "problem_type": (
            training_result[
                "problem_type"
            ]
        ),

        "target_column": (
            training_result[
                "target_column"
            ]
        ),

        "best_model": (
            training_result[
                "best_model_name"
            ]
        ),

        "best_score": (
            training_result[
                "best_score"
            ]
        ),

        "results": (
            training_result[
                "results"
            ]
        ),

        # Actual trained pipeline
        "trained_model": trained_model,

        # Explainability
        "feature_importance": (
            feature_importance
        ),

        "top_features": (
            top_features
        ),
    }


# ============================================================
# MODEL RESULTS DATAFRAME
# ============================================================

def get_model_results_dataframe(
    ml_result: dict,
) -> pd.DataFrame:
    """
    Convert model results into a DataFrame
    for Streamlit.
    """

    results = ml_result.get(
        "results",
        [],
    )

    if not results:

        return pd.DataFrame()

    dataframe = pd.DataFrame(
        results
    )

    rename_map = {

        "model": "Model",

        "metric": "Metric",

        "score": "Score",

        "r2": "R²",

        "mae": "MAE",

        "rmse": "RMSE",

        "accuracy": "Accuracy",

        "error": "Error",
    }

    dataframe = dataframe.rename(
        columns=rename_map
    )

    return dataframe


# ============================================================
# BEST MODEL SUMMARY
# ============================================================

def get_best_model_summary(
    ml_result: dict,
) -> dict:
    """
    Create a business-friendly summary
    of the best model.
    """

    problem_type = (
        ml_result[
            "problem_type"
        ]
    )

    best_model = (
        ml_result[
            "best_model"
        ]
    )

    best_score = (
        ml_result[
            "best_score"
        ]
    )

    if (
        problem_type
        == "classification"
    ):

        metric = "Accuracy"

    else:

        metric = "R²"

    return {

        "model": best_model,

        "problem_type": problem_type,

        "metric": metric,

        "score": float(
            best_score
        ),
    }