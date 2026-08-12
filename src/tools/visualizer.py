import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# HELPERS
# ==================================================

def format_label(value):
    """
    Convert technical names into readable labels.

    Example:
        product_category -> Product Category
        SUM(revenue)     -> Sum(Revenue)
    """

    text = str(value)

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    return text.title()


def format_number(value):
    """
    Format numbers for business-friendly display.
    """

    try:

        value = float(value)

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"

        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"

        return f"{value:,.2f}"

    except (ValueError, TypeError):

        return str(value)


def is_numeric(series):
    """
    Check whether a pandas Series is numeric.
    """

    return pd.api.types.is_numeric_dtype(series)


# ==================================================
# BAR CHART
# ==================================================

def create_bar_chart(
    dataframe,
    x_column,
    y_column,
):

    if dataframe.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    x_values = (
        dataframe[x_column]
        .astype(str)
    )

    y_values = pd.to_numeric(
        dataframe[y_column],
        errors="coerce",
    )

    bars = axis.bar(
        x_values,
        y_values,
    )

    axis.set_xlabel(
        format_label(x_column)
    )

    axis.set_ylabel(
        format_label(y_column)
    )

    axis.set_title(
        f"{format_label(y_column)} by "
        f"{format_label(x_column)}",
        fontsize=14,
        fontweight="bold",
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    for bar, value in zip(
        bars,
        y_values,
    ):

        if pd.isna(value):
            continue

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            format_number(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    return figure


# ==================================================
# HORIZONTAL BAR CHART
# ==================================================

def create_horizontal_bar_chart(
    dataframe,
    x_column,
    y_column,
):

    if dataframe.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    values = pd.to_numeric(
        dataframe[y_column],
        errors="coerce",
    )

    labels = (
        dataframe[x_column]
        .astype(str)
    )

    bars = axis.barh(
        labels,
        values,
    )

    axis.set_xlabel(
        format_label(y_column)
    )

    axis.set_ylabel(
        format_label(x_column)
    )

    axis.set_title(
        f"{format_label(y_column)} by "
        f"{format_label(x_column)}",
        fontsize=14,
        fontweight="bold",
    )

    for bar, value in zip(
        bars,
        values,
    ):

        if pd.isna(value):
            continue

        axis.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {format_number(value)}",
            va="center",
            fontsize=9,
        )

    figure.tight_layout()

    return figure


# ==================================================
# LINE CHART
# ==================================================

def create_line_chart(
    dataframe,
    x_column,
    y_column,
):

    if dataframe.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    values = pd.to_numeric(
        dataframe[y_column],
        errors="coerce",
    )

    x_values = (
        dataframe[x_column]
        .astype(str)
    )

    axis.plot(
        x_values,
        values,
        marker="o",
    )

    axis.set_xlabel(
        format_label(x_column)
    )

    axis.set_ylabel(
        format_label(y_column)
    )

    axis.set_title(
        f"{format_label(y_column)} Trend",
        fontsize=14,
        fontweight="bold",
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    for index, value in enumerate(
        values
    ):

        if pd.isna(value):
            continue

        axis.annotate(
            format_number(value),
            (
                index,
                value,
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    figure.tight_layout()

    return figure


# ==================================================
# PIE CHART
# ==================================================

def create_pie_chart(
    dataframe,
    label_column,
    value_column,
):

    if dataframe.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(7, 7)
    )

    values = pd.to_numeric(
        dataframe[value_column],
        errors="coerce",
    )

    labels = (
        dataframe[label_column]
        .astype(str)
    )

    axis.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
    )

    axis.set_title(
        f"{format_label(value_column)} Distribution",
        fontsize=14,
        fontweight="bold",
    )

    return figure


# ==================================================
# AUTOMATIC CHART SELECTION
# ==================================================

def create_automatic_chart(
    dataframe,
):
    """
    Automatically determine the best visualization.

    Returns:

        figure, chart_type

    chart_type can be:

        single_value
        bar
        horizontal_bar
        line
        None
    """

    # ------------------------------------------------
    # Validate dataframe
    # ------------------------------------------------

    if dataframe is None:
        return None, None

    if dataframe.empty:
        return None, None

    # ------------------------------------------------
    # Only visualize two-column results
    # ------------------------------------------------

    if len(dataframe.columns) != 2:
        return None, None

    x_column = dataframe.columns[0]
    y_column = dataframe.columns[1]

    # =================================================
    # IMPORTANT:
    # SINGLE ROW RESULT
    # =================================================

    # Example:
    #
    # year | SUM(revenue)
    # 2021 | 361232
    #
    # This should be a KPI, not a chart.

    if len(dataframe) == 1:

        return None, "single_value"

    # ------------------------------------------------
    # Convert Y column to numeric
    # ------------------------------------------------

    converted = pd.to_numeric(
        dataframe[y_column],
        errors="coerce",
    )

    if converted.notna().sum() == 0:

        return None, None

    dataframe = dataframe.copy()

    dataframe[y_column] = converted

    # ------------------------------------------------
    # Detect date-like X column
    # ------------------------------------------------

    converted_dates = pd.to_datetime(
        dataframe[x_column],
        errors="coerce",
    )

    date_ratio = (
        converted_dates.notna().sum()
        / len(dataframe)
    )

    # ------------------------------------------------
    # Time series
    # ------------------------------------------------

    if date_ratio >= 0.70:

        chart = create_line_chart(
            dataframe,
            x_column,
            y_column,
        )

        return chart, "line"

    # ------------------------------------------------
    # Small categorical dataset
    # ------------------------------------------------

    if len(dataframe) <= 6:

        chart = create_bar_chart(
            dataframe,
            x_column,
            y_column,
        )

        return chart, "bar"

    # ------------------------------------------------
    # Larger categorical dataset
    # ------------------------------------------------

    chart = create_horizontal_bar_chart(
        dataframe,
        x_column,
        y_column,
    )

    return chart, "horizontal_bar"