import re

import numpy as np
import pandas as pd


# ==================================================
# NORMALIZE COLUMN NAME
# ==================================================

def normalize_column_name(
    column_name: str,
) -> str:
    """
    Normalize a column name for comparison.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column_name).lower(),
    )


# ==================================================
# DIRECT TARGET NAME MATCHING
# ==================================================

def find_suspicious_target_names(
    dataframe: pd.DataFrame,
    target_column: str,
) -> list:
    """
    Find columns whose names strongly suggest that
    they may represent the same business quantity
    as the target.

    Example:

        Target = Revenue

        Suspicious:
            Revenue_Amount
            Total_Revenue
            Revenue_Value
    """

    target_normalized = (
        normalize_column_name(
            target_column
        )
    )

    suspicious = []

    for column in dataframe.columns:

        if column == target_column:
            continue

        normalized = (
            normalize_column_name(
                column
            )
        )

        if (
            target_normalized
            and
            (
                target_normalized in normalized
                or normalized in target_normalized
            )
        ):

            suspicious.append(
                {
                    "column": column,
                    "reason": (
                        "Column name is strongly "
                        "related to the target."
                    ),
                }
            )

    return suspicious


# ==================================================
# CORRELATION-BASED DETECTION
# ==================================================

def find_high_correlations(
    dataframe: pd.DataFrame,
    target_column: str,
    threshold: float = 0.95,
) -> list:
    """
    Find numeric features that have extremely
    high correlation with the target.

    High correlation alone does NOT prove leakage.
    It is treated as a warning signal.
    """

    suspicious = []

    if target_column not in dataframe.columns:

        return suspicious

    target = dataframe[
        target_column
    ]

    if not pd.api.types.is_numeric_dtype(
        target
    ):

        return suspicious

    numeric_columns = (
        dataframe.select_dtypes(
            include=np.number
        ).columns
    )

    for column in numeric_columns:

        if column == target_column:

            continue

        try:

            correlation = (
                dataframe[
                    [
                        column,
                        target_column,
                    ]
                ]
                .corr()
                .iloc[0, 1]
            )

        except Exception:

            continue

        if pd.isna(
            correlation
        ):

            continue

        if abs(correlation) >= threshold:

            suspicious.append(
                {
                    "column": column,
                    "correlation": float(
                        correlation
                    ),
                    "reason": (
                        "Extremely high correlation "
                        "with the target."
                    ),
                }
            )

    return suspicious


# ==================================================
# DERIVED BUSINESS RELATIONSHIPS
# ==================================================

def detect_derived_relationships(
    dataframe: pd.DataFrame,
    target_column: str,
) -> list:
    """
    Detect common mathematical relationships.

    Examples:

        Revenue ≈ Unit_Price × Order_Quantity

        Revenue ≈ Cost + Profit

        Profit ≈ Revenue - Cost

    These relationships are especially important
    for business datasets.
    """

    warnings = []

    if target_column not in dataframe.columns:

        return warnings

    normalized_target = (
        normalize_column_name(
            target_column
        )
    )

    columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    # ==================================================
    # REVENUE
    # ==================================================

    if normalized_target == "revenue":

        # ----------------------------------------------
        # Unit Price × Order Quantity
        # ----------------------------------------------

        if (
            "unitprice" in columns
            and
            "orderquantity" in columns
        ):

            warnings.append(
                {
                    "columns": [
                        columns["unitprice"],
                        columns["orderquantity"],
                    ],
                    "reason": (
                        "Revenue may be directly derived "
                        "from Unit Price × Order Quantity."
                    ),
                }
            )

        # ----------------------------------------------
        # Cost + Profit
        # ----------------------------------------------

        if (
            "cost" in columns
            and
            "profit" in columns
        ):

            warnings.append(
                {
                    "columns": [
                        columns["cost"],
                        columns["profit"],
                    ],
                    "reason": (
                        "Revenue may be directly derived "
                        "from Cost + Profit."
                    ),
                }
            )

    # ==================================================
    # PROFIT
    # ==================================================

    if normalized_target == "profit":

        if (
            "revenue" in columns
            and
            "cost" in columns
        ):

            warnings.append(
                {
                    "columns": [
                        columns["revenue"],
                        columns["cost"],
                    ],
                    "reason": (
                        "Profit may be directly derived "
                        "from Revenue - Cost."
                    ),
                }
            )

    # ==================================================
    # COST
    # ==================================================

    if normalized_target == "cost":

        if (
            "revenue" in columns
            and
            "profit" in columns
        ):

            warnings.append(
                {
                    "columns": [
                        columns["revenue"],
                        columns["profit"],
                    ],
                    "reason": (
                        "Cost may be directly derived "
                        "from Revenue - Profit."
                    ),
                }
            )

    return warnings


# ==================================================
# COMPLETE LEAKAGE ANALYSIS
# ==================================================

def analyze_target_leakage(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Perform a complete target leakage analysis.
    """

    if dataframe is None:

        raise ValueError(
            "No dataframe was provided."
        )

    if dataframe.empty:

        raise ValueError(
            "The dataframe is empty."
        )

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    name_warnings = (
        find_suspicious_target_names(
            dataframe,
            target_column,
        )
    )

    correlation_warnings = (
        find_high_correlations(
            dataframe,
            target_column,
        )
    )

    derived_warnings = (
        detect_derived_relationships(
            dataframe,
            target_column,
        )
    )

    total_warnings = (
        len(name_warnings)
        +
        len(correlation_warnings)
        +
        len(derived_warnings)
    )

    if total_warnings == 0:

        risk_level = "low"

    elif total_warnings <= 2:

        risk_level = "medium"

    else:

        risk_level = "high"

    return {
        "target_column": target_column,

        "risk_level": risk_level,

        "has_warning": (
            total_warnings > 0
        ),

        "name_warnings": name_warnings,

        "correlation_warnings": (
            correlation_warnings
        ),

        "derived_warnings": (
            derived_warnings
        ),

        "total_warnings": (
            total_warnings
        ),
    }