import sys
from pathlib import Path

import pandas as pd


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ==================================================
# INSIGHTAI IMPORTS
# ==================================================

from src.ml.service import (
    get_dataset_profile,
    get_target_information,
    get_model_results_dataframe,
    get_best_model_summary,
)


# ==================================================
# TEST DATASET
# ==================================================

dataframe = pd.DataFrame(
    {
        "Age": [
            21, 25, 30, 35, 40,
            22, 28, 33, 38, 45,
            24, 27, 31, 36, 42,
            23, 29, 34, 39, 44,
        ],

        "Income": [
            25000, 30000, 35000, 40000, 50000,
            27000, 32000, 37000, 45000, 55000,
            28000, 33000, 38000, 43000, 52000,
            26000, 34000, 39000, 47000, 58000,
        ],

        "PurchaseAmount": [
            1200, 1500, 1800, 2200, 3000,
            1300, 1700, 2100, 2500, 3500,
            1400, 1600, 2000, 2400, 3200,
            1250, 1750, 2150, 2700, 3700,
        ],
    }
)


# ==================================================
# PROFILE TEST
# ==================================================

print()
print("=" * 60)
print("INSIGHTAI ML SERVICE TEST")
print("=" * 60)


print()
print("1. DATASET PROFILE")
print("-" * 60)

profile = get_dataset_profile(
    dataframe
)

print(
    f"Rows: {profile['rows']}"
)

print(
    f"Columns: {profile['columns']}"
)

print(
    f"Numeric: "
    f"{profile['numeric_columns']}"
)

print(
    f"Categorical: "
    f"{profile['categorical_columns']}"
)


# ==================================================
# TARGET TEST
# ==================================================

print()
print("2. TARGET ANALYSIS")
print("-" * 60)

target = get_target_information(
    dataframe,
    "PurchaseAmount",
)

print(
    f"Target: "
    f"{target['target_column']}"
)

print(
    f"Problem type: "
    f"{target['problem_type']}"
)

print(
    f"Data type: "
    f"{target['data_type']}"
)

print(
    f"Unique values: "
    f"{target['unique_values']}"
)

print(
    f"Mean: "
    f"{target['mean']:.2f}"
)


# ==================================================
# MODEL RESULT HELPERS
# ==================================================

print()
print("3. MODEL RESULT HELPERS")
print("-" * 60)

fake_result = {
    "problem_type": "regression",
    "best_model": "random_forest",
    "best_score": 0.91,
    "results": [
        {
            "model": "linear_regression",
            "metric": "r2",
            "score": 0.72,
        },
        {
            "model": "random_forest",
            "metric": "r2",
            "score": 0.91,
        },
    ],
}


results_dataframe = (
    get_model_results_dataframe(
        fake_result
    )
)

print(
    results_dataframe
)


summary = (
    get_best_model_summary(
        fake_result
    )
)

print()
print(
    f"Best model: "
    f"{summary['model']}"
)

print(
    f"Metric: "
    f"{summary['metric']}"
)

print(
    f"Score: "
    f"{summary['score']:.4f}"
)


print()
print("=" * 60)
print("ML SERVICE TEST PASSED")
print("=" * 60)