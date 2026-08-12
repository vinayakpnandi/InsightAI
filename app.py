import os
import tempfile

import pandas as pd
import streamlit as st

from src.rag.ingest import ingest_pdf
from src.rag.qa import answer_question

from src.database.loader import load_dataset
from src.database.sql_agent import ask_sql_agent

from src.tools.visualizer import (
    create_automatic_chart,
    format_label,
    format_number,
)

from src.ml.service import (
    get_target_information,
    run_ml_analysis,
    get_model_results_dataframe,
    get_best_model_summary,
)

from src.ml.prediction import (
    get_prediction_features,
    get_feature_metadata,
    make_prediction,
)

from src.ml.business_insights import (
    generate_business_insights,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="InsightAI",
    page_icon="🧠",
    layout="wide",
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def prepare_dataframe_for_ml(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the uploaded DataFrame for ML.

    Handles:
    - whitespace in column names
    - currency values
    - comma-separated numbers
    - numeric strings

    Example:

        "$1,266.00 " -> 1266.0
        "1,250"      -> 1250.0
    """

    if dataframe is None:

        return dataframe

    dataframe = dataframe.copy()

    # ------------------------------------------------
    # Clean column names
    # ------------------------------------------------

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------
    # Clean individual columns
    # ------------------------------------------------

    for column in dataframe.columns:

        series = dataframe[column]

        # Only attempt conversion for object/string
        # columns.
        if (
            series.dtype == "object"
            or pd.api.types.is_string_dtype(series)
        ):

            cleaned = (
                series
                .astype(str)
                .str.strip()
            )

            # ----------------------------------------
            # Handle currency / formatted numbers
            # ----------------------------------------

            numeric_candidate = (
                cleaned
                .str.replace(
                    ",",
                    "",
                    regex=False,
                )
                .str.replace(
                    "$",
                    "",
                    regex=False,
                )
                .str.replace(
                    "₹",
                    "",
                    regex=False,
                )
                .str.replace(
                    "€",
                    "",
                    regex=False,
                )
                .str.replace(
                    "£",
                    "",
                    regex=False,
                )
                .str.replace(
                    "%",
                    "",
                    regex=False,
                )
            )

            # Handle accounting negatives:
            # (1,250.00) -> -1250.00

            negative_mask = (
                numeric_candidate
                .str.match(
                    r"^\(.*\)$",
                    na=False,
                )
            )

            numeric_candidate = (
                numeric_candidate
                .str.replace(
                    "(",
                    "",
                    regex=False,
                )
                .str.replace(
                    ")",
                    "",
                    regex=False,
                )
            )

            converted = pd.to_numeric(
                numeric_candidate,
                errors="coerce",
            )

            # ----------------------------------------
            # Only convert when most non-empty values
            # are actually numeric.
            # ----------------------------------------

            non_empty = (
                cleaned
                .replace(
                    {
                        "": pd.NA,
                        "nan": pd.NA,
                        "None": pd.NA,
                    }
                )
                .notna()
                .sum()
            )

            valid_numeric = (
                converted.notna().sum()
            )

            if (
                non_empty > 0
                and valid_numeric / non_empty >= 0.80
            ):

                converted = converted.copy()

                converted.loc[
                    negative_mask
                ] = -converted.loc[
                    negative_mask
                ].abs()

                dataframe[column] = converted

    return dataframe


def read_uploaded_dataset(
    file_path: str,
    extension: str,
) -> pd.DataFrame:
    """
    Read CSV/XLSX/XLS file into pandas.
    """

    if extension == ".csv":

        dataframe = pd.read_csv(
            file_path
        )

    elif extension in [
        ".xlsx",
        ".xls",
    ]:

        dataframe = pd.read_excel(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported dataset format."
        )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    return dataframe


# ==================================================
# HEADER
# ==================================================

st.title("🧠 InsightAI")

st.subheader(
    "Enterprise AI Decision Intelligence Platform"
)

st.write(
    "Upload documents or business datasets and "
    "ask questions using natural language."
)

st.divider()


# ==================================================
# SESSION STATE
# ==================================================

# --------------------------------------------------
# Document state
# --------------------------------------------------

if "document_processed" not in st.session_state:

    st.session_state.document_processed = False


if "document_id" not in st.session_state:

    st.session_state.document_id = None


if "document_name" not in st.session_state:

    st.session_state.document_name = None


if "page_count" not in st.session_state:

    st.session_state.page_count = 0


if "chunk_count" not in st.session_state:

    st.session_state.chunk_count = 0


# --------------------------------------------------
# Dataset state
# --------------------------------------------------

if "dataset_processed" not in st.session_state:

    st.session_state.dataset_processed = False


if "dataset_name" not in st.session_state:

    st.session_state.dataset_name = None


if "dataset_rows" not in st.session_state:

    st.session_state.dataset_rows = 0


if "dataset_columns" not in st.session_state:

    st.session_state.dataset_columns = []


if "dataset_numeric_columns" not in st.session_state:

    st.session_state.dataset_numeric_columns = []


if "dataset_categorical_columns" not in st.session_state:

    st.session_state.dataset_categorical_columns = []


if "dataset_quality" not in st.session_state:

    st.session_state.dataset_quality = None


if "dataset_preview" not in st.session_state:

    st.session_state.dataset_preview = None


if "dataset_dataframe" not in st.session_state:

    st.session_state.dataset_dataframe = None


# --------------------------------------------------
# ML state
# --------------------------------------------------

if "ml_result" not in st.session_state:

    st.session_state.ml_result = None


if "ml_target_column" not in st.session_state:

    st.session_state.ml_target_column = None


if "business_insights" not in st.session_state:

    st.session_state.business_insights = None

if "latest_prediction" not in st.session_state:

    st.session_state.latest_prediction = None

if "latest_prediction_inputs" not in st.session_state:

    st.session_state.latest_prediction_inputs = None


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📁 Data Sources")

    # ==================================================
    # PDF UPLOAD
    # ==================================================

    st.subheader("📄 Documents")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    process_pdf = st.button(
        "Process PDF",
        type="primary",
        use_container_width=True,
    )

    # ==================================================
    # DATASET UPLOAD
    # ==================================================

    st.subheader("📊 Datasets")

    uploaded_dataset = st.file_uploader(
        "Upload CSV or Excel",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        key="dataset_uploader",
    )

    process_dataset = st.button(
        "Process Dataset",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    # ==================================================
    # ACTIVE SOURCES
    # ==================================================

    st.subheader("📌 Active Sources")

    # --------------------------------------------------
    # PDF
    # --------------------------------------------------

    if st.session_state.document_processed:

        st.success(
            "PDF ready"
        )

        st.caption(
            f"📄 {st.session_state.document_name}"
        )

        st.caption(
            f"Pages: "
            f"{st.session_state.page_count}"
        )

        st.caption(
            f"Chunks: "
            f"{st.session_state.chunk_count}"
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    if st.session_state.dataset_processed:

        st.success(
            "Dataset ready"
        )

        st.caption(
            f"📊 {st.session_state.dataset_name}"
        )

        st.caption(
            f"Rows: "
            f"{st.session_state.dataset_rows}"
        )

        st.caption(
            f"Columns: "
            f"{len(st.session_state.dataset_columns)}"
        )

        quality = (
            st.session_state.dataset_quality
        )

        if quality:

            st.caption(
                f"Missing values: "
                f"{quality['total_missing_values']}"
            )

            st.caption(
                f"Duplicate rows: "
                f"{quality['duplicate_rows']}"
            )


# ==================================================
# PDF SELECTION
# ==================================================

if uploaded_pdf is not None:

    st.info(
        f"Selected PDF: "
        f"**{uploaded_pdf.name}**"
    )


# ==================================================
# PDF PROCESSING
# ==================================================

if process_pdf:

    if uploaded_pdf is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        with st.spinner(
            "Processing PDF..."
        ):

            temp_path = None

            try:

                # --------------------------------------
                # Create temporary file
                # --------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        uploaded_pdf.getbuffer()
                    )

                    temp_path = temp_file.name

                # --------------------------------------
                # Ingest
                # --------------------------------------

                result = ingest_pdf(
                    file_path=temp_path,
                    document_name=uploaded_pdf.name,
                )

                # --------------------------------------
                # Save state
                # --------------------------------------

                st.session_state.document_processed = True

                st.session_state.document_id = (
                    result["document_id"]
                )

                st.session_state.document_name = (
                    result["document_name"]
                )

                st.session_state.page_count = (
                    result["page_count"]
                )

                st.session_state.chunk_count = (
                    result["chunk_count"]
                )

                st.success(
                    "PDF processed successfully."
                )

            except Exception as error:

                st.error(
                    f"PDF processing failed: {error}"
                )

            finally:

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# ==================================================
# DATASET SELECTION
# ==================================================

if uploaded_dataset is not None:

    st.info(
        f"Selected dataset: "
        f"**{uploaded_dataset.name}**"
    )


# ==================================================
# DATASET PROCESSING
# ==================================================

if process_dataset:

    if uploaded_dataset is None:

        st.warning(
            "Please upload a CSV or Excel file first."
        )

    else:

        with st.spinner(
            "Cleaning, analyzing and loading dataset..."
        ):

            temp_path = None

            try:

                # --------------------------------------
                # File extension
                # --------------------------------------

                extension = os.path.splitext(
                    uploaded_dataset.name
                )[1].lower()

                # --------------------------------------
                # Temporary file
                # --------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension,
                ) as temp_file:

                    temp_file.write(
                        uploaded_dataset.getbuffer()
                    )

                    temp_path = temp_file.name

                # --------------------------------------
                # Load into SQLite
                # --------------------------------------

                result = load_dataset(
                    file_path=temp_path,
                    table_name="uploaded_data",
                )

                # --------------------------------------
                # Read original dataset
                # --------------------------------------

                raw_dataframe = (
                    read_uploaded_dataset(
                        temp_path,
                        extension,
                    )
                )

                # --------------------------------------
                # Prepare ML dataframe
                # --------------------------------------

                ml_dataframe = (
                    prepare_dataframe_for_ml(
                        raw_dataframe
                    )
                )

                # --------------------------------------
                # Store DataFrame for ML
                # --------------------------------------

                st.session_state.dataset_dataframe = (
                    ml_dataframe
                )

                # --------------------------------------
                # Dataset metadata
                # --------------------------------------

                st.session_state.dataset_processed = True

                st.session_state.dataset_name = (
                    uploaded_dataset.name
                )

                st.session_state.dataset_rows = (
                    result["rows"]
                )

                st.session_state.dataset_columns = (
                    result["columns"]
                )

                st.session_state.dataset_numeric_columns = (
                    result.get(
                        "numeric_columns",
                        [],
                    )
                )

                st.session_state.dataset_categorical_columns = (
                    result.get(
                        "categorical_columns",
                        [],
                    )
                )

                st.session_state.dataset_quality = (
                    result.get(
                        "quality_report",
                        None,
                    )
                )

                # --------------------------------------
                # Reset previous ML analysis
                # --------------------------------------

                st.session_state.ml_result = None

                st.session_state.ml_target_column = None

                st.session_state.business_insights = None

                st.session_state.latest_prediction = None

                st.session_state.latest_prediction_inputs = None

                # --------------------------------------
                # Preview
                # --------------------------------------

                st.session_state.dataset_preview = (
                    raw_dataframe.head(10)
                )

                st.success(
                    "Dataset processed successfully."
                )

            except Exception as error:

                st.error(
                    f"Dataset processing failed: {error}"
                )

            finally:

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# ==================================================
# DATASET OVERVIEW
# ==================================================

if st.session_state.dataset_processed:

    st.header("📊 Dataset Overview")

    # ==================================================
    # METRICS
    # ==================================================

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Rows",
            st.session_state.dataset_rows,
        )

    with col2:

        st.metric(
            "Columns",
            len(
                st.session_state.dataset_columns
            ),
        )

    with col3:

        quality = (
            st.session_state.dataset_quality
        )

        missing = (
            quality["total_missing_values"]
            if quality
            else 0
        )

        st.metric(
            "Missing Values",
            missing,
        )

    with col4:

        quality = (
            st.session_state.dataset_quality
        )

        duplicates = (
            quality["duplicate_rows"]
            if quality
            else 0
        )

        st.metric(
            "Duplicate Rows",
            duplicates,
        )

    # ==================================================
    # PREVIEW
    # ==================================================

    if (
        st.session_state.dataset_preview
        is not None
    ):

        with st.expander(
            "👀 Preview Dataset",
            expanded=True,
        ):

            st.dataframe(
                st.session_state.dataset_preview,
                use_container_width=True,
            )

    # ==================================================
    # COLUMNS
    # ==================================================

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.subheader(
            "🔢 Numeric Columns"
        )

        numeric_columns = (
            st.session_state
            .dataset_numeric_columns
        )

        if numeric_columns:

            for column in numeric_columns:

                st.write(
                    f"• `{column}`"
                )

        else:

            st.caption(
                "No numeric columns detected."
            )

    with col2:

        st.subheader(
            "🏷️ Categorical Columns"
        )

        categorical_columns = (
            st.session_state
            .dataset_categorical_columns
        )

        if categorical_columns:

            for column in categorical_columns:

                st.write(
                    f"• `{column}`"
                )

        else:

            st.caption(
                "No categorical columns detected."
            )

    # ==================================================
    # DATA QUALITY
    # ==================================================

    quality = (
        st.session_state.dataset_quality
    )

    if quality:

        st.subheader(
            "🔎 Data Quality"
        )

        if (
            quality["total_missing_values"] == 0
            and quality["duplicate_rows"] == 0
        ):

            st.success(
                "Good — no missing values or "
                "duplicate rows detected."
            )

        else:

            st.warning(
                "This dataset needs attention."
            )

            missing_values = (
                quality["missing_values"]
            )

            missing_columns = {
                column: count
                for column, count
                in missing_values.items()
                if count > 0
            }

            if missing_columns:

                st.write(
                    "**Missing values by column:**"
                )

                st.json(
                    missing_columns
                )

            if quality["duplicate_rows"] > 0:

                st.write(
                    f"Duplicate rows: "
                    f"{quality['duplicate_rows']}"
                )

    st.divider()


# ==================================================
# ML INTELLIGENCE
# ==================================================

if (
    st.session_state.dataset_processed
    and
    st.session_state.dataset_dataframe
    is not None
):

    st.header(
        "🤖 ML Intelligence"
    )

    st.write(
        "Use machine learning to identify "
        "patterns and predict a selected target."
    )

    # ==================================================
    # TARGET SELECTION
    # ==================================================

    dataframe_for_ml = (
        st.session_state.dataset_dataframe
    )

    available_columns = (
        dataframe_for_ml.columns.tolist()
    )

    if available_columns:

        target_column = st.selectbox(
            "🎯 Select prediction target",
            available_columns,
            key="ml_target_column_select",
        )

        # Save selected target
        st.session_state.ml_target_column = (
            target_column
        )

        # ==================================================
        # TARGET ANALYSIS
        # ==================================================

        try:

            target_info = (
                get_target_information(
                    dataframe_for_ml,
                    target_column,
                )
            )

            st.subheader(
                "🔍 Target Analysis"
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Problem Type",
                    target_info[
                        "problem_type"
                    ].title(),
                )

            with col2:

                st.metric(
                    "Data Type",
                    target_info[
                        "data_type"
                    ],
                )

            with col3:

                st.metric(
                    "Unique Values",
                    target_info[
                        "unique_values"
                    ],
                )

            with col4:

                st.metric(
                    "Missing Values",
                    target_info[
                        "missing_values"
                    ],
                )

            # ------------------------------------------------
            # Regression information
            # ------------------------------------------------

            if (
                target_info[
                    "problem_type"
                ]
                == "regression"
            ):

                st.info(
                    f"📈 `{target_column}` has been "
                    f"identified as a **regression target**."
                )

                # Numeric statistics
                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    st.metric(
                        "Minimum",
                        format_number(
                            target_info[
                                "minimum"
                            ]
                        ),
                    )

                with col2:

                    st.metric(
                        "Average",
                        format_number(
                            target_info[
                                "mean"
                            ]
                        ),
                    )

                with col3:

                    st.metric(
                        "Maximum",
                        format_number(
                            target_info[
                                "maximum"
                            ]
                        ),
                    )

            # ------------------------------------------------
            # Classification information
            # ------------------------------------------------

            else:

                st.info(
                    f"🏷️ `{target_column}` has been "
                    f"identified as a **classification target**."
                )

                if (
                    "class_distribution"
                    in target_info
                ):

                    with st.expander(
                        "View Class Distribution"
                    ):

                        st.json(
                            target_info[
                                "class_distribution"
                            ]
                        )

            # ==================================================
            # TRAIN BUTTON
            # ==================================================

            train_button = st.button(
                "🚀 Train & Compare Models",
                type="primary",
                use_container_width=True,
                key="train_ml_models",
            )

            if train_button:

                with st.spinner(
                    "Training ML models... "
                    "This may take a moment."
                ):

                    try:

                        ml_result = (
                            run_ml_analysis(
                                dataframe_for_ml,
                                target_column,
                            )
                        )

                        st.session_state.ml_result = (
                            ml_result
                        )

                        st.success(
                            "ML analysis completed successfully."
                        )

                    except Exception as error:

                        st.error(
                            f"ML training failed: {error}"
                        )

        except Exception as error:

            st.error(
                f"Unable to analyze target: {error}"
            )


# ==================================================
# ML RESULTS
# ==================================================

if (
    st.session_state.ml_result
    is not None
):

    ml_result = (
        st.session_state.ml_result
    )

    st.divider()

    st.header(
        "🏆 ML Model Results"
    )

    # ==================================================
    # BEST MODEL
    # ==================================================

    try:

        summary = (
            get_best_model_summary(
                ml_result
            )
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Best Model",
                summary["model"]
                .replace(
                    "_",
                    " ",
                )
                .title(),
            )

        with col2:

            st.metric(
                "Problem Type",
                summary[
                    "problem_type"
                ].title(),
            )

        with col3:

            st.metric(
                summary["metric"],
                f"{summary['score']:.4f}",
            )

    except Exception as error:

        st.warning(
            f"Unable to display model summary: "
            f"{error}"
        )

    # ==================================================
    # MODEL COMPARISON
    # ==================================================

    st.subheader(
        "📊 Model Comparison"
    )

    try:

        results_dataframe = (
            get_model_results_dataframe(
                ml_result
            )
        )

        if not results_dataframe.empty:

            st.dataframe(
                results_dataframe,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------
            # Performance chart
            # ------------------------------------------

            if (
                "Model"
                in results_dataframe.columns
                and
                "Score"
                in results_dataframe.columns
            ):

                chart_dataframe = (
                    results_dataframe[
                        [
                            "Model",
                            "Score",
                        ]
                    ]
                    .dropna(
                        subset=[
                            "Score"
                        ]
                    )
                    .copy()
                )

                if not chart_dataframe.empty:

                    st.subheader(
                        "📈 Model Performance"
                    )

                    st.bar_chart(
                        chart_dataframe.set_index(
                            "Model"
                        )[
                            "Score"
                        ]
                    )

        else:

            st.info(
                "No model comparison results "
                "are available."
            )

    except Exception as error:

        st.warning(
            f"Unable to display model comparison: "
            f"{error}"
        )

# ==================================================
# TARGET LEAKAGE & EXPLAINABILITY
# ==================================================

if (
    st.session_state.ml_result
    is not None
):

    ml_result = st.session_state.ml_result

    st.divider()
    st.header("🔎 Model Explainability")

    leakage = ml_result.get("leakage", {})

    st.subheader("⚠️ Target Leakage Detection")

    if not leakage:
        st.info("Leakage analysis is not available for this model.")
    else:
        risk_level = leakage.get("risk_level", "low")
        total_warnings = leakage.get("total_warnings", 0)

        if risk_level == "high":
            st.error(
                f"🔴 High leakage risk detected — {total_warnings} warning(s). "
                "Model performance may be artificially inflated."
            )
        elif risk_level == "medium":
            st.warning(
                f"🟠 Medium leakage risk detected — {total_warnings} warning(s). "
                "Review the suspicious features before trusting the model score."
            )
        else:
            st.success("🟢 Low leakage risk — no major leakage signals were detected.")

        name_warnings = leakage.get("name_warnings", [])
        if name_warnings:
            with st.expander("Column-name warnings"):
                for warning in name_warnings:
                    st.write(
                        f"• `{warning.get('column', '')}` — {warning.get('reason', '')}"
                    )

        correlation_warnings = leakage.get("correlation_warnings", [])
        if correlation_warnings:
            with st.expander("High-correlation warnings"):
                for warning in correlation_warnings:
                    st.write(
                        f"• `{warning.get('column', '')}` — "
                        f"correlation: **{warning.get('correlation', 0):.3f}** — "
                        f"{warning.get('reason', '')}"
                    )

        derived_warnings = leakage.get("derived_warnings", [])
        if derived_warnings:
            with st.expander("Potentially derived variables"):
                for warning in derived_warnings:
                    columns = warning.get("columns", [])
                    columns_text = ", ".join(f"`{column}`" for column in columns)
                    st.write(
                        f"• {columns_text} — {warning.get('reason', '')}"
                    )

        if total_warnings > 0:
            st.caption(
                "Leakage detection provides warning signals; it does not prove that leakage exists. "
                "Review the data-generating process before using the model for decisions."
            )

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    st.subheader("🔎 Feature Importance")

    top_features = ml_result.get("top_features")
    feature_importance = ml_result.get("feature_importance")

    if top_features is not None and not top_features.empty:
        display_features = top_features.copy()

        display_features = display_features.rename(
            columns={
                "feature": "Feature",
                "importance": "Importance",
                "importance_percent": "Importance (%)",
            }
        )

        display_columns = [
            column for column in ["Feature", "Importance", "Importance (%)"]
            if column in display_features.columns
        ]

        if display_columns:
            table = display_features[display_columns].copy()

            if "Importance" in table.columns:
                table["Importance"] = table["Importance"].round(4)

            if "Importance (%)" in table.columns:
                table["Importance (%)"] = table["Importance (%)"].round(2)

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

        if (
            "Feature" in display_features.columns
            and "Importance (%)" in display_features.columns
        ):
            chart_source = (
                display_features[["Feature", "Importance (%)"]]
                .dropna()
                .sort_values("Importance (%)", ascending=True)
            )

            if not chart_source.empty:
                st.subheader("📊 Feature Importance Chart")
                st.bar_chart(
                    chart_source.set_index("Feature")["Importance (%)"]
                )

    elif feature_importance is not None and not feature_importance.empty:
        st.info("Feature importance was calculated, but no top-feature table is available.")
    else:
        st.info(
            "Feature importance is not available for the selected model. "
            "This can happen when the model does not expose feature importance or coefficients."
        )


# ==================================================
# INTERACTIVE PREDICTION
# ==================================================

if st.session_state.ml_result is not None:

    st.divider()
    st.header("🎯 Interactive Prediction")

    ml_result = st.session_state.ml_result
    trained_model = ml_result.get("trained_model")
    target_column = ml_result.get("target_column")
    prediction_dataframe = st.session_state.dataset_dataframe
    problem_type = ml_result.get("problem_type", "classification")

    if trained_model is None:
        st.warning("Train a model before making predictions.")

    elif prediction_dataframe is None:
        st.warning(
            "The processed dataset is not available. "
            "Please process the dataset again."
        )

    elif target_column not in prediction_dataframe.columns:
        st.warning(
            "The selected target column could not be found "
            "in the current dataset."
        )

    else:

        best_model_name = ml_result.get("best_model", "best model")

        if problem_type == "regression":
            prediction_mode_text = "numeric value"
        else:
            prediction_mode_text = "category/class"

        st.write(
            f"Use the trained **"
            f"{str(best_model_name).replace('_', ' ').title()}** "
            f"model to predict **{target_column}** as a "
            f"**{prediction_mode_text}**."
        )

        if problem_type == "regression":
            st.info(
                "📈 Regression mode: enter the feature values "
                "and InsightAI will estimate the numeric target."
            )
        else:
            st.info(
                "🏷️ Classification mode: enter the feature values "
                "and InsightAI will predict the target class."
            )

        try:

            feature_metadata = get_feature_metadata(
                prediction_dataframe,
                target_column,
            )

            feature_columns = [
                feature["name"]
                for feature in feature_metadata
            ]

            if not feature_metadata:

                st.info(
                    "No prediction features are available "
                    "after excluding the target column."
                )

            else:

                prediction_values = {}

                st.subheader("Enter Prediction Values")

                for feature in feature_metadata:

                    name = feature["name"]

                    safe_key = (
                        str(name)
                        .replace(" ", "_")
                        .replace(".", "_")
                        .replace("/", "_")
                        .replace("-", "_")
                        .replace("$", "")
                        .replace("₹", "")
                        .replace(",", "")
                    )

                    if feature["type"] == "numeric":

                        minimum = feature.get("minimum", 0.0)
                        maximum = feature.get("maximum", 0.0)
                        mean = feature.get("mean", 0.0)

                        if not pd.notna(minimum):
                            minimum = 0.0

                        if not pd.notna(maximum):
                            maximum = minimum

                        if not pd.notna(mean):
                            mean = minimum

                        minimum = float(minimum)
                        maximum = float(maximum)
                        mean = float(mean)

                        if minimum == maximum:

                            prediction_values[name] = minimum

                            st.number_input(
                                name,
                                value=minimum,
                                disabled=True,
                                key=f"predict_{safe_key}",
                            )

                        else:

                            prediction_values[name] = st.number_input(
                                name,
                                min_value=minimum,
                                max_value=maximum,
                                value=mean,
                                key=f"predict_{safe_key}",
                            )

                    else:

                        values = feature.get("values", [])

                        if values:

                            prediction_values[name] = st.selectbox(
                                name,
                                values,
                                key=f"predict_{safe_key}",
                            )

                        else:

                            prediction_values[name] = st.text_input(
                                name,
                                key=f"predict_{safe_key}",
                            )

                if st.button(
                    "🎯 Make Prediction",
                    type="primary",
                    use_container_width=True,
                    key="make_prediction_button",
                ):

                    try:

                        prediction = make_prediction(
                            trained_model,
                            prediction_values,
                            feature_columns,
                        )

                        if hasattr(prediction, "item"):
                            try:
                                prediction = prediction.item()
                            except Exception:
                                pass

                        st.session_state.latest_prediction = prediction
                        st.session_state.latest_prediction_inputs = prediction_values

                        st.success(
                            "Prediction generated successfully."
                        )

                        st.subheader(
                            f"🎯 Predicted {target_column}"
                        )

                        if problem_type == "regression":

                            numeric_prediction = float(prediction)

                            currency_targets = {
                                "revenue",
                                "sales",
                                "profit",
                                "cost",
                                "price",
                                "amount",
                                "income",
                                "salary",
                                "unit_cost",
                                "unit_price",
                                "total",
                            }

                            target_lower = str(
                                target_column
                            ).strip().lower()

                            if target_lower in currency_targets:
                                display_value = f"${numeric_prediction:,.2f}"
                            else:
                                display_value = f"{numeric_prediction:,.2f}"

                            st.metric(
                                label=target_column,
                                value=display_value,
                            )

                            st.caption(
                                "This is a model estimate, "
                                "not a guaranteed business outcome."
                            )

                        else:

                            st.metric(
                                label=target_column,
                                value=str(prediction),
                            )

                        with st.expander("🔍 View Prediction Inputs"):

                            st.dataframe(
                                pd.DataFrame([prediction_values]),
                                use_container_width=True,
                                hide_index=True,
                            )

                        summary = get_best_model_summary(ml_result)

                        with st.expander("📊 Model Performance"):

                            st.write(
                                f"**Best model:** "
                                f"{str(summary['model']).replace('_', ' ').title()}"
                            )

                            if problem_type == "regression":

                                st.write(
                                    f"**R²:** {summary['score']:.4f}"
                                )

                                results = ml_result.get("results", [])

                                best_result = next(
                                    (
                                        item
                                        for item in results
                                        if item.get("model") == summary["model"]
                                    ),
                                    None,
                                )

                                if best_result:

                                    metric_cols = st.columns(2)

                                    with metric_cols[0]:

                                        if best_result.get("mae") is not None:
                                            st.metric(
                                                "MAE",
                                                f"{float(best_result['mae']):,.2f}",
                                            )

                                    with metric_cols[1]:

                                        if best_result.get("rmse") is not None:
                                            st.metric(
                                                "RMSE",
                                                f"{float(best_result['rmse']):,.2f}",
                                            )

                            else:

                                st.write(
                                    f"**Accuracy:** {summary['score']:.4f}"
                                )

                    except Exception as error:

                        st.error(
                            f"Prediction failed: {error}"
                        )

        except Exception as error:

            st.error(
                f"Unable to create prediction inputs: {error}"
            )


# ==================================================
# AI BUSINESS DECISION ENGINE
# ==================================================

if st.session_state.ml_result is not None:

    st.divider()
    st.header("🧠 AI Business Decision Engine")

    st.write(
        "Use Qwen3:4b to turn the ML analysis and prediction "
        "into concise, business-oriented insights, risks, and recommendations."
    )

    latest_prediction = st.session_state.get(
        "latest_prediction",
        None,
    )

    latest_prediction_inputs = st.session_state.get(
        "latest_prediction_inputs",
        None,
    )

    if latest_prediction is not None:

        st.info(
            f"A prediction of **{latest_prediction}** is available. "
            "The AI explanation will include it."
        )

    else:

        st.caption(
            "No prediction has been generated yet. "
            "The AI will still analyze the model, feature importance, "
            "and leakage results."
        )

    generate_business_button = st.button(
        "🧠 Generate AI Business Insights",
        type="primary",
        use_container_width=True,
        key="generate_business_insights_button",
    )

    if generate_business_button:

        with st.spinner(
            "Qwen3:4b is generating business insights... "
            "This may take a few seconds."
        ):

            try:

                business_result = generate_business_insights(
                    ml_result=st.session_state.ml_result,
                    prediction=latest_prediction,
                    prediction_inputs=latest_prediction_inputs,
                )

                st.session_state.business_insights = (
                    business_result
                )

            except Exception as error:

                st.session_state.business_insights = None

                st.error(
                    f"Business insight generation failed: {error}"
                )

    business_result = st.session_state.get(
        "business_insights",
        None,
    )

    if business_result:

        st.subheader("📊 Executive Summary")

        executive_summary = business_result.get(
            "executive_summary",
            "",
        )

        if executive_summary:

            st.info(
                executive_summary
            )

        # --------------------------------------------------
        # Key insights
        # --------------------------------------------------

        key_insights = business_result.get(
            "key_insights",
            [],
        )

        st.subheader("🔍 Key Insights")

        if key_insights:

            for insight in key_insights:

                st.write(
                    f"• {insight}"
                )

        else:

            st.caption(
                "No key insights were returned."
            )

        # --------------------------------------------------
        # Risks
        # --------------------------------------------------

        risks = business_result.get(
            "risks",
            [],
        )

        st.subheader("⚠️ Business Risks")

        if risks:

            for risk in risks:

                st.warning(
                    risk
                )

        else:

            st.success(
                "No additional business risks were identified by the AI."
            )

        # --------------------------------------------------
        # Recommendations
        # --------------------------------------------------

        recommendations = business_result.get(
            "recommendations",
            [],
        )

        st.subheader("💡 Recommendations")

        if recommendations:

            for index, recommendation in enumerate(
                recommendations,
                start=1,
            ):

                st.write(
                    f"**{index}.** {recommendation}"
                )

        else:

            st.caption(
                "No recommendations were returned."
            )

        # --------------------------------------------------
        # Prediction interpretation
        # --------------------------------------------------

        prediction_interpretation = business_result.get(
            "prediction_interpretation",
            "",
        )

        st.subheader("🎯 Prediction Interpretation")

        if prediction_interpretation:

            st.write(
                prediction_interpretation
            )

        else:

            st.caption(
                "No prediction interpretation was returned."
            )

        # --------------------------------------------------
        # Raw structured response
        # --------------------------------------------------

        with st.expander(
            "🔍 View Structured AI Response"
        ):

            st.json(
                business_result
            )


# ==================================================
# ASK INSIGHTAI
# ==================================================

st.divider()

st.header(
    "💬 Ask InsightAI"
)

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: Which region generated "
        "the highest revenue?"
    ),
)


# ==================================================
# KNOWLEDGE SOURCE
# ==================================================

source_type = st.radio(
    "Choose knowledge source",
    [
        "📄 Document",
        "📊 Dataset",
    ],
    horizontal=True,
)


# ==================================================
# ASK BUTTON
# ==================================================

if st.button(
    "Ask InsightAI",
    type="primary",
    use_container_width=True,
):

    # ==================================================
    # VALIDATE QUESTION
    # ==================================================

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    # ==================================================
    # DOCUMENT / RAG
    # ==================================================

    elif source_type == "📄 Document":

        if not st.session_state.document_processed:

            st.warning(
                "Please upload and process "
                "a PDF first."
            )

        else:

            with st.spinner(
                "InsightAI is analyzing the document..."
            ):

                try:

                    result = answer_question(
                        question=question,
                        document_id=(
                            st.session_state.document_id
                        ),
                    )

                    # ----------------------------------
                    # Answer
                    # ----------------------------------

                    st.subheader(
                        "🤖 InsightAI Answer"
                    )

                    st.write(
                        result["answer"]
                    )

                    # ----------------------------------
                    # Sources
                    # ----------------------------------

                    if result.get(
                        "sources"
                    ):

                        st.subheader(
                            "📚 Sources"
                        )

                        for source in (
                            result["sources"]
                        ):

                            st.write(
                                f"**"
                                f"{source['document_name']}"
                                f"** — Page "
                                f"{source['page']}"
                            )

                except Exception as error:

                    st.error(
                        f"RAG error: {error}"
                    )

    # ==================================================
    # DATASET / SQL
    # ==================================================

    elif source_type == "📊 Dataset":

        if not st.session_state.dataset_processed:

            st.warning(
                "Please upload and process "
                "a CSV or Excel dataset first."
            )

        else:

            with st.spinner(
                "InsightAI is analyzing your dataset..."
            ):

                try:

                    # ----------------------------------
                    # SQL Agent
                    # ----------------------------------

                    result = ask_sql_agent(
                        question=question,
                        table_name="uploaded_data",
                    )

                    # ----------------------------------
                    # Business Answer
                    # ----------------------------------

                    st.subheader(
                        "🤖 InsightAI Answer"
                    )

                    st.write(
                        result["answer"]
                    )

                    # ----------------------------------
                    # Query Result
                    # ----------------------------------

                    if result.get("rows"):

                        st.subheader(
                            "📊 Query Result"
                        )

                        result_dataframe = (
                            pd.DataFrame(
                                result["rows"],
                                columns=result["columns"],
                            )
                        )

                        st.dataframe(
                            result_dataframe,
                            use_container_width=True,
                        )

                        # ----------------------------------
                        # Automatic Visualization
                        # ----------------------------------

                        chart, chart_type = (
                            create_automatic_chart(
                                result_dataframe
                            )
                        )

                        # ----------------------------------
                        # Single value
                        # ----------------------------------

                        if (
                            chart_type
                            == "single_value"
                        ):

                            value = (
                                result_dataframe
                                .iloc[0, 1]
                            )

                            label = (
                                result_dataframe
                                .columns[1]
                            )

                            st.subheader(
                                "📌 Key Result"
                            )

                            st.metric(
                                format_label(
                                    label
                                ),
                                format_number(
                                    value
                                ),
                            )

                        # ----------------------------------
                        # Chart
                        # ----------------------------------

                        elif chart is not None:

                            st.subheader(
                                "📈 Visualization"
                            )

                            st.pyplot(
                                chart,
                                use_container_width=True,
                            )

                            import matplotlib.pyplot as plt

                            plt.close(
                                chart
                            )

                        else:

                            st.caption(
                                "No suitable visualization "
                                "could be generated for this "
                                "query result."
                            )

                    # ----------------------------------
                    # Generated SQL
                    # ----------------------------------

                    with st.expander(
                        "🔍 View Generated SQL"
                    ):

                        st.code(
                            result["sql"],
                            language="sql",
                        )

                except Exception as error:

                    st.error(
                        f"SQL analysis error: {error}"
                    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "🧠 InsightAI — Enterprise AI "
    "Decision Intelligence Platform"
)