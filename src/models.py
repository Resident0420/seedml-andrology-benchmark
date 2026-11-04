from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.benchmark import BalancedSampleWeightClassifier


def logistic_model(preprocessor):
    """
    Multiclass logistic regression baseline.

    Uses balanced class weights because SEED-ML
    contains substantial class imbalance.
    """

    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def random_forest_model(preprocessor):
    """
    Random Forest multiclass classifier.
    """

    classifier = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def xgboost_model(preprocessor):

    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. "
            "Install it with: pip install xgboost"
        )

    classifier = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )

    weighted_classifier = BalancedSampleWeightClassifier(
        estimator=classifier
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", weighted_classifier)
        ]
    )

    return model


def lightgbm_model(preprocessor):
    """
    LightGBM multiclass classifier.

    Requires:
        pip install lightgbm
    """

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError(
            "LightGBM is not installed. "
            "Install it with: pip install lightgbm"
        )

    classifier = LGBMClassifier(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbose=-1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def catboost_model(preprocessor):
    """
    CatBoost multiclass classifier.

    Requires:
        pip install catboost
    """

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError(
            "CatBoost is not installed. "
            "Install it with: pip install catboost"
        )

    classifier = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=1,
        verbose=0
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def mlp_model(preprocessor):
    """
    Multilayer Perceptron neural network baseline.
    """

    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    weighted_classifier = BalancedSampleWeightClassifier(
        estimator=classifier
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", weighted_classifier),
        ]
    )

    return model

def get_models(preprocessor):
    """
    Return dictionary containing benchmark models.

    Models requiring optional packages are added only
    if their corresponding libraries are installed.
    """

    models = {
        "Logistic Regression": logistic_model(preprocessor),
        "Random Forest": random_forest_model(preprocessor),
        "MLP": mlp_model(preprocessor)
    }

    try:
        models["XGBoost"] = xgboost_model(preprocessor)
    except ImportError:
        pass

    try:
        models["LightGBM"] = lightgbm_model(preprocessor)
    except ImportError:
        pass

    try:
        models["CatBoost"] = catboost_model(preprocessor)
    except ImportError:
        pass

    return models

def get_search_spaces():
    return {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
        },
        "Random Forest": {
            "classifier__n_estimators": [300, 500, 1000],
            "classifier__max_features": ["sqrt", "log2", 0.5],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5],
        },
        "XGBoost": {
            "classifier__estimator__n_estimators": [300, 500, 1000],
            "classifier__estimator__learning_rate": [0.03, 0.05, 0.1],
            "classifier__estimator__max_depth": [3, 6, 9],
            "classifier__estimator__min_child_weight": [1, 5, 10],
            "classifier__estimator__subsample": [0.8, 1.0],
            "classifier__estimator__colsample_bytree": [0.8, 1.0],
        },
        "LightGBM": {
            "classifier__n_estimators": [300, 500, 1000],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__num_leaves": [15, 31, 63],
            "classifier__min_child_samples": [5, 10, 20],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
        "CatBoost": {
            "classifier__iterations": [300, 500, 1000],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__depth": [4, 6, 8],
            "classifier__l2_leaf_reg": [1, 3, 10],
        },
        "MLP": {
            "classifier__estimator__hidden_layer_sizes": [
                (64,),
                (128,),
                (128, 64),
            ],
            "classifier__estimator__alpha": [0.0001, 0.001, 0.01],
            "classifier__estimator__learning_rate_init": [0.0003, 0.001],
        },
    }