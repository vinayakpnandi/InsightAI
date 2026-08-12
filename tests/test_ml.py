import sys
from pathlib import Path

import pandas as pd


# ==================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# INSIGHTAI IMPORTS
# ==================================================

from src.ml.profiler import profile_dataset
from src.ml.predictor import train_model


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset and standardize
    the column names.
    """

    # Remove accidental quotation marks
    file_path = file_path.strip().strip('"').strip("'")

    path = Path(file_path)

    # ----------------------------------------------
    # Check File
    # ----------------------------------------------

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    extension = path.suffix.lower()

    # ----------------------------------------------
    # Load CSV
    # ----------------------------------------------

    if extension == ".csv":

        dataframe = pd.read_csv(
            path
        )

    # ----------------------------------------------
    # Load Excel
    # ----------------------------------------------

    elif extension in [".xlsx", ".xls"]:

        dataframe = pd.read_excel(
            path
        )

    # ----------------------------------------------
    # Unsupported File
    # ----------------------------------------------

    else:

        raise ValueError(
            "Unsupported file type.\n"
            "Please provide a CSV or Excel file."
        )

    # ----------------------------------------------
    # Clean Column Names
    # ----------------------------------------------

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    # ----------------------------------------------
    # Remove Completely Empty Rows
    # ----------------------------------------------

    dataframe = dataframe.dropna(
        how="all"
    )

    # ----------------------------------------------
    # Validate Dataset
    # ----------------------------------------------

    if dataframe.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )

    return dataframe


# ==================================================
# DISPLAY DATASET INFORMATION
# ==================================================

def display_dataset_information(
    dataframe: pd.DataFrame,
):
    """
    Display basic information about the dataset.
    """

    print()
    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    print()
    print("Column names:")

    for index, column in enumerate(
        dataframe.columns,
        start=1,
    ):

        print(
            f"  {index}. {column}"
        )


# ==================================================
# DISPLAY PROFILE
# ==================================================

def display_profile(
    profile: dict,
):
    """
    Display the dataset profile.
    """

    print()
    print("=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)

    print(
        f"Rows: {profile['rows']}"
    )

    print(
        f"Columns: {profile['columns']}"
    )

    print()

    print(
        "Numeric columns:"
    )

    for column in profile[
        "numeric_columns"
    ]:

        print(
            f"  - {column}"
        )

    print()

    print(
        "Categorical columns:"
    )

    for column in profile[
        "categorical_columns"
    ]:

        print(
            f"  - {column}"
        )

    print()

    print(
        f"Duplicate rows: "
        f"{profile['duplicate_rows']}"
    )

    print()

    print(
        "Missing values:"
    )

    for column, count in profile[
        "missing_values"
    ].items():

        if count > 0:

            print(
                f"  - {column}: {count}"
            )

    # If there are no missing values
    if not any(
        count > 0
        for count in profile[
            "missing_values"
        ].values()
    ):

        print(
            "  No missing values"
        )


# ==================================================
# DISPLAY TARGET INFORMATION
# ==================================================

def display_target_information(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Display information about the selected
    target column.
    """

    target = dataframe[target_column]

    print()
    print("=" * 60)
    print("TARGET INFORMATION")
    print("=" * 60)

    print(
        f"Target column: "
        f"{target_column}"
    )

    print(
        f"Data type: "
        f"{target.dtype}"
    )

    print(
        f"Unique values: "
        f"{target.nunique()}"
    )

    print(
        f"Missing values: "
        f"{target.isnull().sum()}"
    )

    # ----------------------------------------------
    # Numeric Target
    # ----------------------------------------------

    if pd.api.types.is_numeric_dtype(
        target
    ):

        print(
            f"Minimum: "
            f"{target.min()}"
        )

        print(
            f"Maximum: "
            f"{target.max()}"
        )

        print(
            f"Mean: "
            f"{target.mean():.2f}"
        )


# ==================================================
# DISPLAY MODEL RESULTS
# ==================================================

def display_model_results(
    result: dict,
):
    """
    Display ML model training results.
    """

    print()
    print("=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(
        f"Problem type: "
        f"{result['problem_type']}"
    )

    print(
        f"Target column: "
        f"{result['target_column']}"
    )

    print(
        f"Best model: "
        f"{result['best_model_name']}"
    )

    print(
        f"Best score: "
        f"{result['best_score']:.4f}"
    )

    print()

    print(
        "Individual model results:"
    )

    print()

    for model_result in result[
        "results"
    ]:

        print(
            f"Model: "
            f"{model_result['model']}"
        )

        print(
            f"Metric: "
            f"{model_result['metric']}"
        )

        # ------------------------------------------
        # Score
        # ------------------------------------------

        if model_result.get(
            "score"
        ) is not None:

            print(
                f"Score: "
                f"{model_result['score']:.4f}"
            )

        # ------------------------------------------
        # Regression Metrics
        # ------------------------------------------

        if "r2" in model_result:

            print(
                f"R²: "
                f"{model_result['r2']:.4f}"
            )

        if "mae" in model_result:

            print(
                f"MAE: "
                f"{model_result['mae']:.4f}"
            )

        if "rmse" in model_result:

            print(
                f"RMSE: "
                f"{model_result['rmse']:.4f}"
            )

        # ------------------------------------------
        # Error
        # ------------------------------------------

        if "error" in model_result:

            print(
                f"Error: "
                f"{model_result['error']}"
            )

        print(
            "-" * 40
        )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("INSIGHTAI ML TEST")
    print("=" * 60)

    # ==================================================
    # GET FILE PATH
    # ==================================================

    file_path = input(
        "\nEnter CSV/XLSX path: "
    )

    # ==================================================
    # LOAD DATASET
    # ==================================================

    try:

        dataframe = load_dataset(
            file_path
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("DATASET LOADING ERROR")
        print("=" * 60)

        print()
        print(error)

        sys.exit(1)

    # ==================================================
    # DISPLAY DATASET INFORMATION
    # ==================================================

    display_dataset_information(
        dataframe
    )

    # ==================================================
    # PROFILE DATASET
    # ==================================================

    profile = profile_dataset(
        dataframe
    )

    display_profile(
        profile
    )

    # ==================================================
    # GET TARGET COLUMN
    # ==================================================

    print()

    target_column = input(
        "Enter target column: "
    ).strip()

    # ==================================================
    # VALIDATE TARGET COLUMN
    # ==================================================

    if target_column not in dataframe.columns:

        print()
        print("=" * 60)
        print("INVALID TARGET COLUMN")
        print("=" * 60)

        print()
        print(
            f"'{target_column}' "
            "is not a column in the dataset."
        )

        print()
        print(
            "Available columns:"
        )

        for column in dataframe.columns:

            print(
                f"  - {column}"
            )

        sys.exit(1)

    # ==================================================
    # DISPLAY TARGET INFORMATION
    # ==================================================

    display_target_information(
        dataframe,
        target_column,
    )

    # ==================================================
    # TRAIN ML MODELS
    # ==================================================

    print()
    print("=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)

    print()
    print(
        "Please wait..."
    )

    try:

        result = train_model(
            dataframe=dataframe,
            target_column=target_column,
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("ML TRAINING ERROR")
        print("=" * 60)

        print()
        print(error)

        sys.exit(1)

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    display_model_results(
        result
    )

    # ==================================================
    # COMPLETE
    # ==================================================

    print()
    print("=" * 60)
    print("ML TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()