import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

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


from src.ml.prediction import (
    get_prediction_features,
    get_feature_metadata,
    build_prediction_dataframe,
)


# ============================================================
# TEST DATA
# ============================================================

df = pd.DataFrame(
    {
        "Age": [
            20,
            25,
            30,
            35,
            40,
        ],

        "Income": [
            20000,
            30000,
            40000,
            50000,
            60000,
        ],

        "PurchaseAmount": [
            500,
            1000,
            1500,
            2000,
            2500,
        ],
    }
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("INSIGHTAI PREDICTION MODULE TEST")
print("=" * 60)


features = get_prediction_features(
    df,
    "PurchaseAmount",
)

print()
print("Prediction features:")
print(features)


metadata = get_feature_metadata(
    df,
    "PurchaseAmount",
)

print()
print("Feature metadata:")

for item in metadata:

    print(item)


values = {
    "Age": 30,
    "Income": 40000,
}


prediction_df = (
    build_prediction_dataframe(
        values,
        features,
    )
)

print()
print("Prediction DataFrame:")
print(prediction_df)


print()
print("=" * 60)
print("PREDICTION MODULE TEST PASSED")
print("=" * 60)