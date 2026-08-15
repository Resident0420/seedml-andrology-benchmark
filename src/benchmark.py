import time

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================
# Balanced Sample-Weight Wrapper
# ============================================

class BalancedSampleWeightClassifier(
    ClassifierMixin,
    BaseEstimator,
):
    """
    Wrapper for classifiers that support sample_weight
    but do not provide class_weight="balanced".

    Balanced sample weights are calculated independently
    within each training fold during cross-validation.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)

        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y,
        )

        self.estimator_.fit(
            X,
            y,
            sample_weight=sample_weight,
        )

        self.classes_ = self.estimator_.classes_

        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


# ============================================
# Single-Model Benchmark
# ============================================

def benchmark_model(
    name,
    model,
    X,
    y,
    n_splits=5,
    n_repeats=5,
    random_state=42,
):
    """
    Benchmark a classification model using repeated
    stratified k-fold cross-validation.

    Parameters
    ----------
    name : str
        Model name.

    model : sklearn-compatible estimator
        Complete modeling pipeline.

    X : pandas.DataFrame
        Predictor variables.

    y : array-like
        Target labels.

    n_splits : int, default=5
        Number of folds per cross-validation repeat.

    n_repeats : int, default=5
        Number of cross-validation repeats.

    random_state : int, default=42
        Random seed controlling fold generation.

    Returns
    -------
    dict
        Summary of model performance and runtime.
    """

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "weighted_f1": "f1_weighted",
    }

    print("=" * 70)
    print(f"Benchmarking: {name}")
    print("=" * 70)

    start_time = time.perf_counter()

    scores = cross_validate(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    runtime = time.perf_counter() - start_time

    result = {
        "Model": name,
        "Macro F1 Mean": scores["test_macro_f1"].mean(),
        "Macro F1 SD": scores["test_macro_f1"].std(),
        "Balanced Accuracy Mean":
            scores["test_balanced_accuracy"].mean(),
        "Balanced Accuracy SD":
            scores["test_balanced_accuracy"].std(),
        "Weighted F1 Mean":
            scores["test_weighted_f1"].mean(),
        "Weighted F1 SD":
            scores["test_weighted_f1"].std(),
        "Runtime Seconds": runtime,
    }

    print(
        f"Macro F1: "
        f"{result['Macro F1 Mean']:.4f} "
        f"± {result['Macro F1 SD']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{result['Balanced Accuracy Mean']:.4f} "
        f"± {result['Balanced Accuracy SD']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{result['Weighted F1 Mean']:.4f} "
        f"± {result['Weighted F1 SD']:.4f}"
    )

    print(f"Runtime: {runtime:.2f} seconds")
    print()

    return result


# ============================================
# Multi-Model Benchmark
# ============================================

def benchmark_models(
    models,
    X,
    y,
    n_splits=5,
    n_repeats=5,
    random_state=42,
):
    """
    Benchmark multiple classification models and
    rank them according to mean Macro F1 score.
    """

    all_results = []

    for name, model in models.items():
        try:
            result = benchmark_model(
                name=name,
                model=model,
                X=X,
                y=y,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )

            all_results.append(result)

        except Exception as error:
            print("=" * 70)
            print(f"ERROR: {name}")
            print(error)
            print("=" * 70)
            print()

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        results_df = (
            results_df
            .sort_values(
                by="Macro F1 Mean",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return results_df


# ============================================
# Results Utilities
# ============================================

def save_benchmark_results(results_df, output_path):
    """Save benchmark results to CSV."""

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"Benchmark results saved to:\n{output_path}")


def print_benchmark_ranking(results_df):
    """Print model rankings based on mean Macro F1."""

    if results_df.empty:
        print("No benchmark results available.")
        return

    print()
    print("=" * 70)
    print("MODEL BENCHMARK RANKING")
    print("=" * 70)

    for index, row in results_df.iterrows():
        print(
            f"{index + 1}. "
            f"{row['Model']} | "
            f"Macro F1 = "
            f"{row['Macro F1 Mean']:.4f} "
            f"± {row['Macro F1 SD']:.4f}"
        )

    print("=" * 70)