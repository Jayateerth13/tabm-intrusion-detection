TabM for Network Intrusion Detection

Benchmark TabM vs MLP, XGBoost, and Random Forest on NSL-KDD (5-class intrusion detection).

## Project structure

```
tabm-intrusion-detection/
├── data/              # NSL-KDD train/test files (not included in repo)
├── src/               # Training and evaluation code
├── results/           # Generated metrics, models, and plots
├── main.ipynb         # Run the full pipeline here
└── requirements.txt
```

**`data/`** — Download from [Mireu-Lab/NSL-KDD on Hugging Face](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD). This folder must contain the two official split files used by `preprocess.py`:
- `KDDTrain+.txt` — training connections (~126k rows)
- `KDDTest+.txt` — test connections (~22k rows)

**`src/`**
- `preprocess.py` — load data, one-hot encode categoricals (122 features), `StandardScaler`
- `train_baselines.py` — train MLP, XGBoost, Random Forest
- `train_tabm.py` — train TabM (BatchEnsemble, class-weight cap)
- `evaluate.py` — accuracy, macro P/R/F1, per-class reports
- `visualize.py` — confusion matrices, metric bar charts, training loss
- `benchmark_inference.py` — test-set inference timing (CPU)

**`results/`** (created by the notebook) — `summary_metrics.csv`, `*_report.txt`, `inference_timing.csv`, saved models (`.joblib` / `.pt`), and PNG figures.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from Hugging Face into `data/` before running (see above).

## Run

```bash
jupyter notebook main.ipynb
```

Run all cells top to bottom. Training TabM last is intentional (higher memory use).

To re-time inference only (after models exist in `results/`):

```bash
python -m src.benchmark_inference
```
