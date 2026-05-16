"""
benchmark_inference.py
----------------------
Measure wall-clock inference time on the NSL-KDD test set for all four models.

Inference = prediction only (no training). Reports total seconds, ms/sample,
and samples/sec. Results are saved to results/inference_timing.csv.
"""

from __future__ import annotations

import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from tabm import TabM


def _time_sklearn_predict(
    model: Any,
    X: np.ndarray,
    *,
    n_warmup: int = 1,
    n_repeat: int = 5,
) -> dict[str, float]:
    """Time sklearn-compatible model.predict on X."""
    n = len(X)
    for _ in range(n_warmup):
        model.predict(X[: min(512, n)])

    times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)

    mean_s = float(np.mean(times))
    return {
        "total_s": mean_s,
        "ms_per_sample": 1000.0 * mean_s / n,
        "samples_per_s": n / mean_s,
        "std_s": float(np.std(times)),
    }


def _time_tabm_predict(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: torch.device | None = None,
    n_warmup: int = 1,
    n_repeat: int = 5,
) -> dict[str, float]:
    """Time TabM forward pass + argmax on X (full test tensor)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(X)
    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            logits = model(x_num=X_t[: min(512, n)]).mean(dim=1)
            _ = torch.argmax(logits, dim=1)

        times: list[float] = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            logits = model(x_num=X_t).mean(dim=1)
            _ = torch.argmax(logits, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_s = float(np.mean(times))
    return {
        "total_s": mean_s,
        "ms_per_sample": 1000.0 * mean_s / n,
        "samples_per_s": n / mean_s,
        "std_s": float(np.std(times)),
    }


def load_tabm_from_checkpoint(
    weights_path: str,
    n_features: int,
    *,
    k: int = 16,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Rebuild TabM and load state dict from tabm_model.pt."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabM.make(
        n_num_features=n_features,
        cat_cardinalities=None,
        d_out=5,
        n_blocks=3,
        d_block=256,
        dropout=0.1,
        k=k,
        arch_type="tabm",
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def benchmark_inference(
    X_test: np.ndarray,
    *,
    mlp_model: Any | None = None,
    xgb_model: Any | None = None,
    rf_model: Any | None = None,
    tabm_model: torch.nn.Module | None = None,
    results_dir: str = "results",
    n_repeat: int = 5,
) -> pd.DataFrame:
    """
    Benchmark inference on X_test for all models.

    Any model left as None is loaded from results_dir (*.joblib or tabm_model.pt).
    """
    os.makedirs(results_dir, exist_ok=True)
    n_test = len(X_test)
    rows: list[dict[str, Any]] = []

    if mlp_model is None:
        mlp_model = joblib.load(os.path.join(results_dir, "mlp_model.joblib"))
    if xgb_model is None:
        xgb_model = joblib.load(os.path.join(results_dir, "xgb_model.joblib"))
    if rf_model is None:
        rf_model = joblib.load(os.path.join(results_dir, "rf_model.joblib"))
    if tabm_model is None:
        tabm_model = load_tabm_from_checkpoint(
            os.path.join(results_dir, "tabm_model.pt"),
            n_features=X_test.shape[1],
        )

    for name, model in [
        ("MLP", mlp_model),
        ("XGBoost", xgb_model),
        ("RandomForest", rf_model),
    ]:
        stats = _time_sklearn_predict(model, X_test, n_repeat=n_repeat)
        rows.append(
            {
                "Model": name,
                "n_test": n_test,
                "n_features": X_test.shape[1],
                "total_s_mean": round(stats["total_s"], 4),
                "total_s_std": round(stats["std_s"], 4),
                "ms_per_sample": round(stats["ms_per_sample"], 4),
                "samples_per_s": round(stats["samples_per_s"], 1),
            }
        )

    tabm_stats = _time_tabm_predict(tabm_model, X_test, n_repeat=n_repeat)
    rows.append(
        {
            "Model": "TabM",
            "n_test": n_test,
            "n_features": X_test.shape[1],
            "total_s_mean": round(tabm_stats["total_s"], 4),
            "total_s_std": round(tabm_stats["std_s"], 4),
            "ms_per_sample": round(tabm_stats["ms_per_sample"], 4),
            "samples_per_s": round(tabm_stats["samples_per_s"], 1),
        }
    )

    df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "inference_timing.csv")
    df.to_csv(out_path, index=False)
    print(f"Inference timing saved to {out_path}\n")
    print(df.to_string(index=False))
    return df


def main() -> None:
    from src.preprocess import load_and_preprocess

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    _, _, X_test, _ = load_and_preprocess(
        "data/KDDTrain+.txt",
        "data/KDDTest+.txt",
    )
    benchmark_inference(X_test)


if __name__ == "__main__":
    main()
