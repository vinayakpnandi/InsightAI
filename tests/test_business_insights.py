import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.ml.service import (
    run_ml_analysis,
)

from src.ml.business_insights import (
    generate_business_insights,
)


# ============================================================
# LOAD DATA
# ============================================================

file_path = (
    r"C:\Users\VINAYAK\Downloads"
    r"\Bike_Sales_Manipulate_Lab_4.2.7.csv"
)

df = pd.read_csv(
    file_path
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# ML ANALYSIS
# ============================================================

print(
    "\nRunning ML analysis..."
)

result = run_ml_analysis(
    df,
    "Revenue",
)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

print(
    "\nGenerating business insights..."
)

insights = generate_business_insights(
    result
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "INSIGHTAI BUSINESS INTELLIGENCE"
)

print(
    "=" * 60
)

print(
    "\nEXECUTIVE SUMMARY:"
)

print(
    insights[
        "executive_summary"
    ]
)

print(
    "\nKEY INSIGHTS:"
)

for item in insights[
    "key_insights"
]:

    print(
        "-",
        item
    )

print(
    "\nRISKS:"
)

for item in insights[
    "risks"
]:

    print(
        "-",
        item
    )

print(
    "\nRECOMMENDATIONS:"
)

for item in insights[
    "recommendations"
]:

    print(
        "-",
        item
    )

print(
    "\nPREDICTION:"
)

print(
    insights[
        "prediction_interpretation"
    ]
)

print(
    "\n"
    + "=" * 60
)