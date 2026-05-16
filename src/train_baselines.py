"""
train_baselines.py
------------------
Trains three classical/baseline ML models on the NSL-KDD dataset.

The three baselines are:
  1. MLP (sklearn) — same hidden dimensions as TabM's backbone for a structural comparison
  2. XGBoost       — gradient boosted trees, the strongest classical tabular baseline
  3. Random Forest — ensemble of decision trees, robust to feature scale

All models receive the same StandardScaler-normalised features produced by preprocess.py.
No resampling is performed: models train on the naturally imbalanced dataset, which reflects
real IDS conditions where U2R and R2L attacks are far rarer than DoS and Probe events.
"""

import gc
import os
import psutil

# Prevent OpenMP from spawning multiple thread pools.
# On macOS, sklearn (OpenMP) + XGBoost (libomp) loading in the same process
# can trigger a duplicate-library abort. These env vars must be set before
# any of the C extensions are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from xgboost import XGBClassifier


def _mem_mb() -> str:
    proc = psutil.Process(os.getpid())
    return f"{proc.memory_info().rss / 1024**2:.0f} MB"


def _log(msg: str) -> None:
    print(f"[LOG] {msg}  (mem={_mem_mb()})", flush=True)


class _XGBIterLog(xgb.callback.TrainingCallback):
    """Print a one-liner every 40 boosting rounds so we can see progress."""
    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 40 == 0:
            _log(f"  XGBoost round {epoch + 1}/400")
        return False


def train_mlp(X_train, y_train, X_test):
    """
    Train a multi-layer perceptron with architecture matching TabM's backbone.

    Architecture: 3 hidden layers of width 256, ReLU activations, Adam optimizer.
    This mirrors TabM's n_blocks=3, d_block=256 setting so any accuracy gap reflects
    the benefit of BatchEnsemble rather than a difference in model capacity.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 41)
    y_train : np.ndarray, shape (n_train,)
    X_test  : np.ndarray, shape (n_test, 41)

    Returns
    -------
    model : fitted MLPClassifier
    preds : np.ndarray of predicted class indices on X_test
    """
    _log("train_mlp — start")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 256, 256),  # 3 layers, 256 units each — matches TabM backbone
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=50,          # same epoch budget as TabM
        batch_size=256,       # same mini-batch size as TabM
        random_state=42,
        early_stopping=False, # train for the full budget, same as TabM
        verbose=True,         # print loss after each epoch so progress is visible
        # Note: MLPClassifier does not support class_weight or sample_weight.
        # Class imbalance is handled at the TabM, XGBoost, and Random Forest level.
    )
    _log("train_mlp — calling fit()")
    model.fit(X_train, y_train)
    _log("train_mlp — fit() done, calling predict()")
    preds = model.predict(X_test)
    _log("train_mlp — predict() done")

    # Force a full GC cycle to release any OpenMP thread pool state held by
    # sklearn before XGBoost loads its own libomp. On macOS this reduces the
    # risk of a duplicate-library segfault.
    gc.collect()
    _log("train_mlp — gc done, returning")

    return model, preds


def train_xgboost(X_train, y_train, X_test):
    """
    Train an XGBoost gradient boosted tree ensemble.

    XGBoost is typically the strongest classical baseline on tabular data.
    Settings use moderate regularisation (subsample, colsample_bytree) to prevent
    overfitting on the dominant DoS class.

    Note: do NOT pass use_label_encoder=False — that argument was removed in
    XGBoost >= 2.0 and will raise a TypeError.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 41)
    y_train : np.ndarray, shape (n_train,)
    X_test  : np.ndarray, shape (n_test, 41)

    Returns
    -------
    model : fitted XGBClassifier
    preds : np.ndarray of predicted class indices on X_test
    """
    _log("train_xgboost — start")
    _log(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
         f"KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    _log(f"  X_train shape={X_train.shape}  X_test shape={X_test.shape}")

    _log("train_xgboost — creating XGBClassifier")
    model = XGBClassifier(
        n_estimators=400,       # more trees for finer boosting steps
        max_depth=6,
        learning_rate=0.05,     # halved from 0.1 — smaller steps generalise better with more trees
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1,
        verbosity=1,
        callbacks=[_XGBIterLog()],
    )
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    _log("train_xgboost — calling fit() with sample_weight")
    model.fit(X_train, y_train, sample_weight=sample_weight)
    _log("train_xgboost — fit() done, calling predict()")
    preds = model.predict(X_test)
    _log("train_xgboost — predict() done")

    # Remove callbacks before returning so joblib can pickle the model.
    # TrainingCallback instances contain internal C state that is not serialisable.
    model.set_params(callbacks=None)

    gc.collect()
    _log("train_xgboost — returning")
    return model, preds


def train_random_forest(X_train, y_train, X_test):
    """
    Train a Random Forest ensemble of decision trees.

    Random Forest with max_depth=None grows fully unpruned trees, which tends to
    produce the best accuracy on this dataset. The 200-tree ensemble provides
    stable predictions via majority voting.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 41)
    y_train : np.ndarray, shape (n_train,)
    X_test  : np.ndarray, shape (n_test, 41)

    Returns
    -------
    model : fitted RandomForestClassifier
    preds : np.ndarray of predicted class indices on X_test
    """
    _log("train_random_forest — start")
    _log(f"  X_train shape={X_train.shape}  X_test shape={X_test.shape}")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=1,            # single-threaded: avoids further OpenMP conflicts after XGBoost
        verbose=0,
        class_weight='balanced',
    )
    _log("train_random_forest — calling fit()")
    model.fit(X_train, y_train)
    _log("train_random_forest — fit() done, calling predict()")
    preds = model.predict(X_test)
    _log("train_random_forest — predict() done, returning")
    gc.collect()
    return model, preds
