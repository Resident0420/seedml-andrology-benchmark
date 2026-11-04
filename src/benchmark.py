import json
import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================
# Balanced Sample-Weight Wrapper
# ============================================

class BalancedSampleWeightClassifier(ClassifierMixin, BaseEstimator):

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
        "Balanced Accuracy Mean": scores["test_balanced_accuracy"].mean(),
        "Balanced Accuracy SD": scores["test_balanced_accuracy"].std(),
        "Weighted F1 Mean": scores["test_weighted_f1"].mean(),
        "Weighted F1 SD": scores["test_weighted_f1"].std(),
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
    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"Benchmark results saved to:\n{output_path}")


def print_benchmark_ranking(results_df):
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


# ============================================
# Nested Hyperparameter-Tuning Benchmark
# ============================================

def nested_benchmark_models(
    models,
    search_spaces,
    X,
    y,
    outer_splits=5,
    inner_splits=3,
    n_iter=12,
    random_state=42,
    search_jobs=2,
):
    y = np.asarray(y)
    labels = np.unique(y)

    if pd.Series(y).value_counts().min() < outer_splits:
        raise ValueError(
            "Too few observations for outer stratified CV."
        )

    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    outer_partitions = list(outer_cv.split(X, y))
    rows = []

    for name, model in models.items():
        print(
            f"\nTuning and evaluating: {name}",
            flush=True,
        )

        space = search_spaces[name]
        candidates = min(n_iter, len(ParameterGrid(space)))

        for fold, (train_idx, valid_idx) in enumerate(
            outer_partitions,
            start=1,
        ):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            if pd.Series(y_train).value_counts().min() < inner_splits:
                raise ValueError(
                    f"{name}, fold {fold}: too few observations "
                    "for inner stratified CV."
                )

            inner_cv = StratifiedKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=random_state + fold,
            )

            search = RandomizedSearchCV(
                estimator=clone(model),
                param_distributions=space,
                n_iter=candidates,
                scoring="f1_macro",
                cv=inner_cv,
                refit=True,
                n_jobs=search_jobs,
                pre_dispatch=search_jobs,
                random_state=random_state,
                error_score="raise",
                return_train_score=False,
            )

            start = time.perf_counter()

            search.fit(X_train, y_train)
            predictions = search.predict(X_valid)

            elapsed = time.perf_counter() - start

            row = {
                "Model": name,
                "Fold": fold,
                "Macro F1": f1_score(
                    y_valid,
                    predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "Balanced Accuracy": balanced_accuracy_score(
                    y_valid,
                    predictions,
                ),
                "Weighted F1": f1_score(
                    y_valid,
                    predictions,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                ),
                "Runtime Seconds": elapsed,
                "Best Parameters": json.dumps(
                    search.best_params_
                ),
            }

            rows.append(row)

            print(
                f"  Fold {fold}/{outer_splits}: "
                f"Macro F1 = {row['Macro F1']:.4f}",
                flush=True,
            )

    details = pd.DataFrame(rows)

    summary = (
        details.groupby("Model", sort=False)
        .agg(
            **{
                "Macro F1 Mean": ("Macro F1", "mean"),
                "Macro F1 SD": ("Macro F1", "std"),
                "Balanced Accuracy Mean": (
                    "Balanced Accuracy", "mean"
                ),
                "Balanced Accuracy SD": (
                    "Balanced Accuracy", "std"
                ),
                "Weighted F1 Mean": ("Weighted F1", "mean"),
                "Weighted F1 SD": ("Weighted F1", "std"),
                "Runtime Seconds": ("Runtime Seconds", "sum"),
            }
        )
        .reset_index()
        .sort_values(
            "Macro F1 Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary, details