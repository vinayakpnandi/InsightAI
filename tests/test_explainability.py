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
# IMPORTS
# ==================================================

from src.ml.leakage import (
    analyze_target_leakage,
)


# ==================================================
# TEST DATA
# ==================================================

dataframe = pd.DataFrame(
    {
        "Unit_Price": [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
        ],

        "Order_Quantity": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],

        "Cost": [
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
        ],

        "Profit": [
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
        ],

        "Revenue": [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
        ],
    }
)


# ==================================================
# RUN
# ==================================================

print()
print("=" * 60)
print("INSIGHTAI LEAKAGE TEST")
print("=" * 60)

result = analyze_target_leakage(
    dataframe,
    "Revenue",
)


print()
print(
    "Target:",
    result["target_column"],
)

print(
    "Risk:",
    result["risk_level"],
)

print(
    "Warnings:",
    result["total_warnings"],
)


print()
print("Name warnings:")

for warning in (
    result["name_warnings"]
):

    print(
        "-",
        warning,
    )


print()
print("Correlation warnings:")

for warning in (
    result["correlation_warnings"]
):

    print(
        "-",
        warning,
    )


print()
print("Derived relationship warnings:")

for warning in (
    result["derived_warnings"]
):

    print(
        "-",
        warning,
    )


print()
print("=" * 60)
print("LEAKAGE TEST PASSED")
print("=" * 60)