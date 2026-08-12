from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)


def get_classification_models():
    """
    Return a collection of classification models.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }


def get_regression_models():
    """
    Return a collection of regression models.
    """

    return {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }