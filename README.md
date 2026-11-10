# Kumo Walkthrough: Credit Card Default Prediction

A single Jupyter notebook that predicts whether a credit card client will **default on their payment next month**. It compares three classic tree-based models, before and after hyperparameter tuning, with [Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular), a pretrained tabular foundation model from NVIDIA that predicts without any training or tuning on this data.

## Dataset

The notebook uses `credit_card_clients_default` from the [TabArena BeyondArena](https://huggingface.co/datasets/TabArena/BeyondArena) collection on Hugging Face: 30,000 clients, 23 features and 1 target. The data has no missing values.

| Columns | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics |
| `PAY_0`, `PAY_2` – `PAY_6` | Repayment status for each of the last six months (−2 to 0 = paid on time or revolving credit, 1 – 8 = months of payment delay) |
| `BILL_AMT1` – `BILL_AMT6` | Bill amount for each month (negative after an overpayment) |
| `PAY_AMT1` – `PAY_AMT6` | Amount paid for each month |
| `DefaultOnPaymentNextMonth` | Target: `Yes` (22%) or `No` (78%) |

The notebook loads the parquet file directly instead of the whole BeyondArena repository, which is much faster. The download is cached in `dataset/`, so later runs don't download it again.

## What the notebook does

1. **Loads and explores the data.**
2. **Transforms the money columns** (`LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`) with a cube root. The cube root compresses the long right tails and, unlike the square root, also works on negative values.
3. **Splits the data** into 80% training and 20% test rows, stratified by the target.
4. **Trains baseline models** with default settings: CatBoost, Random Forest and AdaBoost.
5. **Tunes each model** with [Hyperopt](https://github.com/hyperopt/hyperopt) (TPE, up to 100 trials, stops early after 20 trials without improvement). Each trial is scored by 3-fold cross-validated ROC AUC on the training set only, so the test set stays untouched.
6. **Runs Kumo-Tabular** with in-context learning: the labeled training rows go in as examples together with the test rows, and the model predicts the test rows in one pass (4 ensemble members).
7. **Compares all seven models** with a score table, charts and confusion matrices, and saves the scores to `results.json`.

Because the classes are imbalanced (always predicting `No` already gives about 78% accuracy), the notebook reports MCC, ROC AUC and per-class recall and precision as well as accuracy.

## Results

Test-set scores from the latest run (`results.json`). "Defaulters caught" is recall on the `Yes` class.

| Model | Accuracy | F1 (weighted) | MCC | ROC AUC | Defaulters caught |
|---|---|---|---|---|---|
| CatBoost (base) | 0.822 | 0.801 | 0.408 | 0.785 | 0.37 |
| Random Forest (base) | 0.817 | 0.796 | 0.390 | 0.760 | 0.36 |
| AdaBoost (base) | 0.821 | 0.795 | 0.396 | 0.773 | 0.33 |
| CatBoost (tuned) | 0.823 | 0.801 | 0.409 | 0.789 | 0.37 |
| Random Forest (tuned) | 0.823 | 0.801 | 0.409 | 0.781 | 0.36 |
| AdaBoost (tuned) | 0.819 | 0.794 | 0.390 | 0.779 | 0.34 |
| **Kumo-Tabular** | **0.824** | **0.804** | **0.418** | **0.794** | **0.38** |

Key findings:

- **Kumo-Tabular scored best on every metric**, with no training or tuning.
- **The differences are small.** All seven models are within 0.007 accuracy and 0.028 MCC of each other.
- **Tuning helped little.** It improved ROC AUC for all three models (most for Random Forest: 0.760 → 0.781), but MCC changed by only −0.006 to +0.019.
- **Most defaulters are missed.** At the default 0.5 probability cutoff, the models catch only 33 – 38% of real defaulters. Lowering the cutoff is the most promising next step if catching defaulters matters more than avoiding false alarms.

The train/test split has no fixed seed, so the scores change slightly on each run.

## Getting started

### Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended). The project pins its dependencies in `uv.lock`.
- An NVIDIA GPU is optional. Without one, everything runs on the CPU (see [Runtime](#runtime)).

### Installation

```bash
git clone <repository-url>
cd kumo-walkthrough
uv sync
```

`uv sync` creates a `.venv/` folder and installs the dependencies from `uv.lock`. PyTorch is installed from the CPU-only wheel index, as configured in `pyproject.toml`. To use a GPU, change the `pytorch-cpu` index in `pyproject.toml` to a CUDA index (for example `https://download.pytorch.org/whl/cu128`) before you run `uv sync`.

### Running the notebook

Open `main.ipynb` in VS Code or JupyterLab and select the `.venv` kernel. Then run all cells from top to bottom.

To use JupyterLab without adding it to the project:

```bash
uv run --with jupyterlab jupyter lab main.ipynb
```

Run the cells in order. Each training cell overwrites the shared `model` and `y_pred` variables, and the cell after it stores the scores.

### Runtime

Approximate times on CPU:

| Step | Time |
|---|---|
| Baseline models | A few seconds to a minute |
| Hyperparameter search | About 3 – 20 minutes per model |
| Kumo-Tabular (4 ensemble members) | About 6 – 7 minutes per member |

With an NVIDIA GPU, Kumo-Tabular is much faster and uses float16 automatically.

## Project structure

```
.
├── main.ipynb       # The full walkthrough: data, models, tuning, evaluation
├── results.json     # Test-set scores of all seven models from the latest run
├── dataset/         # Hugging Face cache of the downloaded parquet file
├── pyproject.toml   # Dependencies and Ruff configuration
└── uv.lock          # Locked dependency versions
```

CatBoost writes its training logs to `catboost_info/`. Git ignores this folder.

## Main dependencies

| Package | Use |
|---|---|
| `datasets` | Loads the data from Hugging Face |
| `pandas`, `matplotlib` | Data handling and charts |
| `catboost`, `scikit-learn` | Classic models and evaluation metrics |
| `hyperopt` | Hyperparameter tuning |
| `structured-data-models` (`sdm`), `torch` | Runs Kumo-Tabular |
| `ruff` | Linting and formatting |

## Code style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The rules are in `pyproject.toml`.

```bash
uv run ruff check main.ipynb
uv run ruff format main.ipynb
```
