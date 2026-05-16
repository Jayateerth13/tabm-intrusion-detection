"""
visualize.py
------------
Generates all result plots and saves them to the results/ directory.

Plots produced:
  1. Normalised confusion matrix — one per model (4 total)
  2. Per-class F1 score bar chart — all models side-by-side
  3. Overall metric comparison bar chart (accuracy, precision, recall, F1 macro)
  4. TabM training loss curve over epochs
  5. XGBoost top-20 feature importances (by gain)

All figures are saved at 150 dpi as PNG files.
plt.close() is called after each save to release memory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot and save a row-normalised confusion matrix for one model.

    Row normalisation (normalize='true') converts raw counts to recall per class,
    making it easy to see which attack types are misclassified regardless of class size.

    Parameters
    ----------
    y_true      : np.ndarray — true labels
    y_pred      : np.ndarray — predicted labels
    model_name  : str        — used in the plot title and output filename
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        vmin=0.0, vmax=1.0,
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    save_path = f'results/{model_name}_confusion_matrix.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_f1_comparison(all_metrics):
    """
    Bar chart comparing per-class F1 scores across all four models.

    Groups bars by attack class so it is easy to see which model handles
    each class best — particularly important for the rare R2L and U2R classes.

    Parameters
    ----------
    all_metrics : list of metric dicts returned by evaluate.evaluate_all
    """
    n_classes = len(CLASS_NAMES)
    n_models  = len(all_metrics)
    x         = np.arange(n_classes)
    width     = 0.8 / n_models  # distribute bars evenly within each class group

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, m in enumerate(all_metrics):
        f1s = list(m['f1_per_class'])
        # Pad to n_classes in case a model never predicted a class
        f1s += [0.0] * (n_classes - len(f1s))
        ax.bar(x + i * width, f1s, width, label=m['model'])

    ax.set_xlabel('Attack Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score — All Models')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = 'results/f1_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_overall_metrics(all_metrics):
    """
    Bar chart comparing overall accuracy, macro precision, recall, and F1 for all models.

    Parameters
    ----------
    all_metrics : list of metric dicts returned by evaluate.evaluate_all
    """
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1_macro']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 (macro)']
    n_metrics = len(metric_keys)
    n_models  = len(all_metrics)
    x         = np.arange(n_metrics)
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, m in enumerate(all_metrics):
        vals = [m[k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=m['model'])

        # Annotate each bar with its numeric value for easy reading
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=7,
            )

    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison — All Models')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = 'results/overall_metrics_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tabm_loss(train_losses):
    """
    Plot and save the TabM training loss curve (cross-entropy vs epoch).

    A smoothly decreasing curve confirms that the model is converging; a flat or
    erratic curve would suggest a learning-rate or batch-size problem.

    Parameters
    ----------
    train_losses : list of float — average batch loss recorded after each training epoch
    """
    if not train_losses:
        print("  No TabM training losses to plot — skipping.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('TabM Training Loss Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = 'results/tabm_training_loss.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_xgb_importance(xgb_model):
    """
    Plot and save XGBoost's top-20 features ranked by total gain.

    'Gain' measures the average improvement in the loss function brought by
    each feature across all splits that use it — a more informative importance
    metric than frequency or weight.

    Parameters
    ----------
    xgb_model : fitted XGBClassifier
    """
    try:
        import xgboost as xgb
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(
            xgb_model, ax=ax,
            max_num_features=20,
            importance_type='gain',
            title='XGBoost Feature Importance (Top 20 by Gain)',
        )
        plt.tight_layout()
        save_path = 'results/xgb_feature_importance.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  XGBoost importance plot failed ({e}) — skipping.")


def generate_all_plots(y_test, predictions, all_metrics, tabm_losses, xgb_model):
    """
    Generate and save all result visualisations.

    Parameters
    ----------
    y_test       : np.ndarray — ground-truth test labels
    predictions  : dict       — {model_name: y_pred array}
    all_metrics  : list       — metric dicts from evaluate.evaluate_all
    tabm_losses  : list       — per-epoch training loss values from train_tabm
    xgb_model    : fitted XGBClassifier object
    """
    os.makedirs('results', exist_ok=True)

    # One confusion matrix per model
    for model_name, y_pred in predictions.items():
        plot_confusion_matrix(y_test, y_pred, model_name)

    plot_f1_comparison(all_metrics)
    plot_overall_metrics(all_metrics)
    plot_tabm_loss(tabm_losses)
    plot_xgb_importance(xgb_model)

    print("\nAll plots saved to results/")
