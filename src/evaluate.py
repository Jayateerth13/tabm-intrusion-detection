"""
evaluate.py
-----------
Computes and reports evaluation metrics for all trained models.

Metric design decisions for IDS:
  - Macro averaging (not weighted): treats all 5 classes equally, so a model that
    ignores rare U2R attacks does not artificially inflate its score.
  - zero_division=0: models that never predict R2L or U2R would otherwise raise
    a division-by-zero warning in precision/recall; we handle this gracefully.
  - Per-class F1: critical for IDS because global accuracy can be misleading when
    class distributions are severely imbalanced (U2R is < 0.1% of the test set).
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def compute_metrics(y_true, y_pred, model_name):
    """
    Compute a full suite of classification metrics for a single model.

    Parameters
    ----------
    y_true      : np.ndarray — ground-truth class indices
    y_pred      : np.ndarray — predicted class indices
    model_name  : str        — identifier used in printed output and saved files

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, f1_macro,
                    f1_per_class (array), report (str), confusion_matrix (array)
    """
    labels = np.arange(len(CLASS_NAMES))  # [0, 1, 2, 3, 4] — pinned for consistent ordering
    return {
        'model':            model_name,
        'accuracy':         accuracy_score(y_true, y_pred),
        # Macro precision/recall/F1 — each class contributes equally to the average
        'precision':        precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
        'recall':           recall_score(y_true, y_pred,    average='macro', labels=labels, zero_division=0),
        'f1_macro':         f1_score(y_true, y_pred,        average='macro', labels=labels, zero_division=0),
        # Per-class F1 exposes which attack categories a model handles poorly
        'f1_per_class':     f1_score(y_true, y_pred,        average=None,    labels=labels, zero_division=0),
        # Full sklearn classification report with support counts
        'report':           classification_report(
                                y_true, y_pred,
                                labels=labels,
                                target_names=CLASS_NAMES,
                                zero_division=0
                            ),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate_all(y_test, predictions):
    """
    Evaluate all models, print summaries, save per-model reports, and write a CSV.

    Parameters
    ----------
    y_test      : np.ndarray — ground-truth labels for the test set
    predictions : dict       — {model_name (str): y_pred (np.ndarray)}

    Returns
    -------
    dict with keys:
        'metrics'     : list of metric dicts (one per model)
        'predictions' : the same predictions dict passed in
    """
    os.makedirs('results', exist_ok=True)
    all_metrics = []

    for model_name, y_pred in predictions.items():
        m = compute_metrics(y_test, y_pred, model_name)
        all_metrics.append(m)

        # Print a concise summary to stdout
        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f} (macro)")
        print(f"  Recall:    {m['recall']:.4f} (macro)")
        print(f"  F1:        {m['f1_macro']:.4f} (macro)")
        print(f"\nClassification Report:\n{m['report']}")

        # Persist the full classification report as a text file
        report_path = f"results/{model_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy:  {m['accuracy']:.4f}\n")
            f.write(f"Precision: {m['precision']:.4f} (macro)\n")
            f.write(f"Recall:    {m['recall']:.4f} (macro)\n")
            f.write(f"F1:        {m['f1_macro']:.4f} (macro)\n\n")
            f.write(m['report'])

    # --- Summary CSV: one row per model, one column per metric --- #
    rows = []
    for m in all_metrics:
        row = {
            'Model':     m['model'],
            'Accuracy':  round(m['accuracy'],  4),
            'Precision': round(m['precision'], 4),
            'Recall':    round(m['recall'],    4),
            'F1_macro':  round(m['f1_macro'],  4),
        }
        # Append per-class F1 columns; pad with 0.0 if a class is missing from predictions
        for i, cls in enumerate(CLASS_NAMES):
            row[f'F1_{cls}'] = (
                round(float(m['f1_per_class'][i]), 4)
                if i < len(m['f1_per_class'])
                else 0.0
            )
        rows.append(row)

    summary_path = 'results/summary_metrics.csv'
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary metrics saved to {summary_path}")

    return {'metrics': all_metrics, 'predictions': predictions}
