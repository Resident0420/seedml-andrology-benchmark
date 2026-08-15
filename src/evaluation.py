import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted multiclass classification model
    on a held-out test set.
    """

    y_pred = model.predict(X_test)

    results = {
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred
        ),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            average="macro"
        ),
        "weighted_f1": f1_score(
            y_test,
            y_pred,
            average="weighted"
        ),
    }

    print("Classification Report")
    print("=" * 70)
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print(f"Weighted F1:       {results['weighted_f1']:.4f}")

    return results

def get_per_class_metrics(
    model,
    X_test,
    y_test,
    class_names=None,
):
    """
    Calculate per-class precision, recall, F1 score,
    and support for a fitted multiclass classifier.
    """

    y_pred = model.predict(X_test)
    classes = model.classes_

    if class_names is None:
        class_names = classes

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = pd.DataFrame(report).T.loc[
        class_names,
        ["precision", "recall", "f1-score", "support"],
    ]

    per_class_metrics.columns = [
        "Precision",
        "Recall",
        "F1 Score",
        "Support",
    ]

    return per_class_metrics

def cross_validate_model(
    model,
    X,
    y,
    n_splits=5,
    n_repeats=5,
    random_state=42,
):
    """
    Evaluate a model using repeated stratified
    k-fold cross-validation.
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

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = pd.DataFrame({
        "Metric": [
            "Macro F1",
            "Balanced Accuracy",
            "Weighted F1",
        ],
        "Mean": [
            scores["test_macro_f1"].mean(),
            scores["test_balanced_accuracy"].mean(),
            scores["test_weighted_f1"].mean(),
        ],
        "SD": [
            scores["test_macro_f1"].std(),
            scores["test_balanced_accuracy"].std(),
            scores["test_weighted_f1"].std(),
        ],
    })

    return summary

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    class_names=None,
    normalize="true",
):
    """
    Plot a multiclass confusion matrix.

    Parameters
    ----------
    model : fitted classifier
        Trained classification model.

    X_test : pandas.DataFrame
        Test predictors.

    y_test : array-like
        True test labels.

    class_names : list, optional
        Display names for classes.
        If None, model.classes_ is used.

    normalize : {"true", "pred", "all"} or None
        Confusion matrix normalization mode.
    """

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_,
        normalize=normalize,
    )

    display_labels = (
        class_names
        if class_names is not None
        else model.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format=".2f" if normalize else "d",
    )

    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    return fig, ax

def plot_calibration(
    model,
    X_test,
    y_test,
    class_names=None,
    n_bins=10,
):
    """
    Plot one-vs-rest calibration curves
    for multiclass classification.
    """

    y_prob = model.predict_proba(X_test)

    classes = model.classes_

    if class_names is None:
        class_names = classes

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, class_label in enumerate(classes):

        y_binary = (
            np.asarray(y_test) == class_label
        ).astype(int)

        prob_true, prob_pred = calibration_curve(
            y_binary,
            y_prob[:, i],
            n_bins=n_bins,
            strategy="quantile",
        )

        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=class_names[i],
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction Positive")
    ax.set_title("One-vs-Rest Calibration Curves")

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    return fig, ax

def plot_roc(
    model,
    X_test,
    y_test,
    class_names,
    ax=None,
):
    """
    Publication-style one-vs-rest ROC curves for multiclass classification.
    """

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt

    y_score = model.predict_proba(X_test)

    n_classes = len(class_names)

    y_test_bin = label_binarize(
        y_test,
        classes=np.arange(n_classes)
    )

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    roc_auc = {}

    for i, (class_name, color) in enumerate(zip(class_names, colors)):

        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        roc_auc[i] = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            lw=2.2,
            color=color,
            label=f"{class_name}: {roc_auc[i]:.3f}"
        )

    ax.plot(
        [0,1],
        [0,1],
        "--",
        color="gray",
        alpha=0.5,
        linewidth=1,
        label="Chance"
    )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    ax.set_title(
        "A. One-vs-Rest ROC Curves",
        fontsize=13,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(alpha=0.2)

    ax.legend(
        title="Diagnostic class",
        loc="center left",
        bbox_to_anchor=(1.02,0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )

    return fig, ax

def plot_precision_recall(
    model,
    X_test,
    y_test,
    class_names,
    ax=None,
):

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import numpy as np
    import matplotlib.pyplot as plt

    y_score = model.predict_proba(X_test)
    classes = model.classes_

    y_binary = label_binarize(
        y_test,
        classes=classes
    )

    colors = plt.cm.tab10(np.linspace(0,1,len(classes)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    for i, (class_name, color) in enumerate(zip(class_names, colors)):

        precision, recall, _ = precision_recall_curve(
            y_binary[:,i],
            y_score[:,i]
        )

        ap = average_precision_score(
            y_binary[:,i],
            y_score[:,i]
        )

        ax.plot(
            recall,
            precision,
            lw=2.2,
            color=color,
            label=f"{class_name}: {ap:.3f}"
        )

    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    ax.set_title(
        "B. One-vs-Rest Precision–Recall Curves",
        fontsize=13,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(alpha=0.2)

    ax.legend(
        title="Diagnostic class",
        loc="center left",
        bbox_to_anchor=(1.02,0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )

    return fig, ax