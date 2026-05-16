"""
train_tabm.py
-------------
Trains a TabM model (ICLR 2025) on the NSL-KDD intrusion detection dataset.

TabM — Tabular Model with Multiple predictions — is a deep learning architecture
that efficiently represents an ensemble of k MLPs via BatchEnsemble weight sharing.
Each forward pass produces k predictions (one per sub-model); they are averaged at
inference time to obtain a single robust output.

Key architectural details used here:
  - n_blocks=3, d_block=256: matches the MLP baseline depth and width for fair comparison
  - k=16 sub-models: halved from paper default (k=32) for faster CPU training
  - arch_type='tabm': uses full BatchEnsemble (as opposed to the lightweight 'tabm-mini')
  - All 41 NSL-KDD features are passed as numerical inputs (categorical features were
    label-encoded and standardized in the preprocessing step)
  - Class-weighted loss (capped at 50×) to counter R2L/U2R under-representation
  - Cosine annealing LR schedule: decays lr smoothly from 1e-3 to 0 over 50 epochs

Reference: Gorishniy et al., "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
           Ensembling", ICLR 2025. https://arxiv.org/abs/2410.24210
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tabm import TabM


def train_tabm(X_train, y_train, X_test, y_test):
    """
    Build, train, and evaluate a TabM model for 5-class intrusion detection.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 41), float32
    y_train : np.ndarray, shape (n_train,), int64
    X_test  : np.ndarray, shape (n_test, 41), float32
    y_test  : np.ndarray, shape (n_test,), int64   (unused during training — for reference)

    Returns
    -------
    model       : trained TabM model
    preds       : np.ndarray of predicted class indices on X_test
    train_losses: list of average cross-entropy loss per epoch
    """
    # Fix all random sources inside this function for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  TabM training on: {device}")

    # Move data to the selected device as PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    # --- Model instantiation --- #
    # TabM.make() is the recommended factory method — it validates parameters and
    # fills in architecture-specific defaults (e.g., initialisation schemes).
    #
    # n_num_features: all NSL-KDD features after one-hot encoding + scaling (computed dynamically)
    # cat_cardinalities=None: pass None (not []) — no separate categorical embedding needed
    # d_out=5: one logit per class (Normal, DoS, Probe, R2L, U2R)
    # k=32: paper default ensemble size; was previously halved to k=16 to save memory
    n_num_features = X_train.shape[1]  # dynamic: changes with one-hot encoding
    model = TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=None,
        d_out=5,
        n_blocks=3,
        d_block=256,
        dropout=0.1,
        k=16,
        arch_type='tabm',
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total TabM parameters: {total_params:,}")

    # Class weights: inverse frequency so rare classes (R2L, U2R) get higher loss penalty.
    # Weights are capped at 50× the minimum weight. Both R2L (~25×) and U2R (~484×)
    # get capped to ~18.5×, giving them equal emphasis without extreme gradient spikes.
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(5),
        y=y_train,
    )
    weights = np.minimum(weights, weights.min() * 50)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"  Class weights (capped): { {i: f'{w:.2f}' for i, w in enumerate(weights)} }")

    # AdamW with mild weight decay — standard choice for tabular deep learning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)

    # Cosine annealing decays lr smoothly from 1e-3 to ~0 over n_epochs,
    # letting the model fine-tune rare-class boundaries in later epochs.
    n_epochs   = 50
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    batch_size = 256
    n          = len(X_train_t)
    train_losses = []

    # --- Training loop --- #
    for epoch in range(n_epochs):
        model.train()

        # Shuffle training indices each epoch for stochastic gradient descent
        perm       = torch.randperm(n, device=device)
        epoch_loss = 0.0
        steps      = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            X_b = X_train_t[idx]
            y_b = y_train_t[idx]

            optimizer.zero_grad()

            # Forward pass: TabM outputs shape (batch_size, k, d_out) during training.
            # Average across the k sub-model predictions to get (batch_size, d_out)
            # before computing the cross-entropy loss.
            logits = model(x_num=X_b)   # (batch, k=16, 5)
            logits = logits.mean(dim=1) # (batch, 5)

            loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps      += 1

        avg_loss = epoch_loss / steps
        train_losses.append(avg_loss)
        scheduler.step()

        # Log progress every 10 epochs to monitor convergence
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on full training set — used as a convergence sanity check,
                # not as a measure of generalisation
                logits_all = model(x_num=X_train_t).mean(dim=1)
                train_acc  = (torch.argmax(logits_all, dim=1) == y_train_t).float().mean().item()
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch + 1:3d}/{n_epochs} — loss: {avg_loss:.4f} | train acc: {train_acc:.4f} | lr: {lr_now:.2e}")
            model.train()

    # --- Test-set predictions --- #
    model.eval()
    with torch.no_grad():
        # Same averaging used during training: mean over k sub-models, then argmax
        logits_test = model(x_num=X_test_t).mean(dim=1)  # (n_test, 5)
        preds       = torch.argmax(logits_test, dim=1).cpu().numpy()

    # Persist trained weights so they can be reloaded without retraining
    torch.save(model.state_dict(), 'results/tabm_model.pt')
    print("  TabM model weights saved to results/tabm_model.pt")

    # Release GPU/CPU tensors so subsequent sklearn models have full memory available.
    # Without this, the large training tensors remain pinned in memory while MLP and
    # XGBoost run, which can exhaust RAM and crash the Jupyter kernel.
    del X_train_t, y_train_t, X_test_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model, preds, train_losses
