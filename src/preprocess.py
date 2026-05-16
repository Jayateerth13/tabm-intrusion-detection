"""
preprocess.py
-------------
Loads and preprocesses the NSL-KDD dataset for the 5-class intrusion detection task.

The NSL-KDD dataset has 41 features (3 categorical + 38 numerical) plus a label column
and a difficulty score column. Preprocessing steps:
  1. Load raw CSV files (no header)
  2. Drop the difficulty score column (irrelevant to model training)
  3. Map string attack labels to 5 integer classes: Normal, DoS, Probe, R2L, U2R
  4. One-hot encode the 3 categorical features (protocol_type, service, flag)
     — avoids implying a numeric ordering to nominal values
  5. Split into X (features) and y (labels)
  6. Standardize all features using StandardScaler fit on training data only
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# All 43 column names in NSL-KDD (41 features + label + difficulty)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# These three columns contain string values and must be label-encoded before scaling
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Human-readable names for the 5 output classes
CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Maps every known NSL-KDD attack string to one of 5 integer class indices.
# The test set (KDDTest+.txt) contains attack types not seen in training — this
# comprehensive dictionary ensures they are correctly mapped rather than dropped.
ATTACK_MAP = {
    # Class 0: Normal traffic
    'normal': 0,
    # Class 1: Denial of Service — overwhelms resources to block legitimate access
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1,
    'teardrop': 1, 'apache2': 1, 'udpstorm': 1, 'processtable': 1,
    'mailbomb': 1, 'worm': 1,
    # Class 2: Probe — surveillance and reconnaissance attacks
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2,
    'mscan': 2, 'saint': 2,
    # Class 3: Remote to Local (R2L) — unauthorized remote access attempts
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3,
    'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3,
    'xlock': 3, 'xsnoop': 3, 'snmpguess': 3, 'snmpgetattack': 3,
    'httptunnel': 3, 'sendmail': 3, 'named': 3,
    # Class 4: User to Root (U2R) — local privilege escalation attacks
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4,
    'ps': 4, 'sqlattack': 4, 'xterm': 4,
}


def load_and_preprocess(train_path, test_path):
    """
    Load raw NSL-KDD CSV files, encode features, and return scaled numpy arrays.

    Parameters
    ----------
    train_path : str
        Path to KDDTrain+.txt
    test_path : str
        Path to KDDTest+.txt

    Returns
    -------
    X_train : np.ndarray of shape (n_train, n_features), dtype float32
    y_train : np.ndarray of shape (n_train,), dtype int64
    X_test  : np.ndarray of shape (n_test, n_features), dtype float32
    y_test  : np.ndarray of shape (n_test,), dtype int64

    Note: n_features > 41 after one-hot encoding the 3 categorical columns.
    """
    # --- Step 1: Load CSVs --- #
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=COLUMNS)

    # --- Step 2: Drop difficulty score --- #
    # The difficulty column (43rd) rates how hard each sample is to classify.
    # It is a meta-label and not a network feature, so we drop it.
    train_df.drop(columns=['difficulty'], inplace=True)
    test_df.drop(columns=['difficulty'],  inplace=True)

    # --- Step 3: Map string attack labels to integer class indices --- #
    train_df['label'] = train_df['label'].map(ATTACK_MAP)
    test_df['label']  = test_df['label'].map(ATTACK_MAP)

    # Any label not in ATTACK_MAP will become NaN after .map(); warn and default to DoS (1)
    for name, df in [('train', train_df), ('test', test_df)]:
        n_unmapped = df['label'].isna().sum()
        if n_unmapped > 0:
            print(f"WARNING: {n_unmapped} unmapped labels in {name} set — defaulting to class 1 (DoS).")
            df['label'].fillna(1, inplace=True)

    train_df['label'] = train_df['label'].astype(int)
    test_df['label']  = test_df['label'].astype(int)

    # --- Step 4: One-hot encode categorical columns --- #
    # pd.get_dummies replaces each categorical column with binary 0/1 indicator columns.
    # This avoids implying a numeric ordering between nominal values
    # (e.g. tcp=0, udp=1, icmp=2 is meaningless to a neural network).
    # protocol_type: {tcp, udp, icmp}           →  3 indicator columns
    # service: {http, ftp, smtp, ...}            → 66 indicator columns
    # flag: {SF, S0, REJ, ...}                   → 11 indicator columns
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLS, dtype=np.float32)
    test_df  = pd.get_dummies(test_df,  columns=CATEGORICAL_COLS, dtype=np.float32)

    # Align columns: any category present in one split but not the other gets
    # a zero-filled column so both matrices have identical shape.
    all_cols = sorted(set(train_df.columns) | set(test_df.columns))
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df  = test_df.reindex(columns=all_cols,  fill_value=0)

    # --- Step 5: Separate features (X) from labels (y) --- #
    feature_cols = [c for c in train_df.columns if c != 'label']
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int64)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df['label'].values.astype(np.int64)

    # --- Step 6: Standardize features --- #
    # Fit StandardScaler only on training data to prevent data leakage,
    # then apply the same transform to the test set.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Print class distributions so we can observe the imbalance
    print("\nTraining class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (y_train == i).sum()
        pct   = 100 * count / len(y_train)
        print(f"  {name:8s}: {count:6d}  ({pct:.1f}%)")

    print("\nTest class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (y_test == i).sum()
        pct   = 100 * count / len(y_test)
        print(f"  {name:8s}: {count:6d}  ({pct:.1f}%)")

    n_features = X_train.shape[1]
    print(f"\nX_train: {X_train.shape}  |  X_test: {X_test.shape}  |  n_features (after OHE): {n_features}")

    return X_train, y_train, X_test, y_test
