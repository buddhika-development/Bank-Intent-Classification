# BANKING77 Intent Classification

SE4050 – Deep Learning, SLIIT (2026). A comparison of four deep learning models :- MLP, BiLSTM, TextCNN and
DistilBERT  for intent classification on the [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets)
dataset (77 fine-grained banking customer-support intents).

**Group:** IT23153172 (Bandara W.G.B.M.W. - DistilBERT), IT23190634 (Senevirathne G.A.D.D. - TextCNN),
IT23347380 (Kumara D.V.U. - BiLSTM, MLP)

## Dataset

- **Source:** [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) (`banking_data/train.csv`, `test.csv`), mirrored on [Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)
- **Citation:** Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* ACL NLP4ConvAI Workshop. [arXiv:2003.04807](https://arxiv.org/abs/2003.04807)
- **License:** CC-BY-4.0
- 10,003 official train sentences, 3,080 official test sentences, 77 intents. No manual download needed — `src/common/data.py` fetches the CSVs directly from GitHub the first time it runs.

## 1. Prerequisites

- **Python 3.13** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** - install with:
```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
  (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Git**

## 2. Setup

```bash
git clone https://github.com/buddhika-development/Bank-Intent-Classification.git
cd Bank-Intent-Classification
uv sync
```

`uv sync` reads `uv.lock` and installs the exact dependency versions used to produce the results in this repo
(PyTorch 2.14.0, Transformers 5.17.0, scikit-learn 1.9.1, etc. — full list in `requirements.txt`). No separate
`pip install` step is needed.

**GPU:** none of the reported results require a GPU; all four models were trained on CPU-only laptops. If a
CUDA GPU is available, PyTorch will use it automatically - no code changes needed.

## 3. Reproducing the shared setup

Run these once, before any individual model. They only need to be run once per machine later runs reuse the
saved files.

```bash
# 1. Build the shared stratified train/val/test split (seed 42, 10% validation)
uv run python -m src.common.data
# -> writes data/splits/{train,val,test}.csv and labels.json

# 2. (Optional) Explore the dataset
# open notebooks/01_eda.ipynb in VS Code or Jupyter and run all cells
```

### GloVe embeddings (needed by MLP, BiLSTM and TextCNN - not DistilBERT)

Two options are used across the codebase, either works and gives the same vectors:

- **MLP and BiLSTM** load a local file at `data/glove/glove.6B.100d.txt`. If it is not already there, download
  it once:
```powershell
  Invoke-WebRequest -Uri "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip" -OutFile "glove.6B.zip"
  Expand-Archive glove.6B.zip -DestinationPath glove_temp
  mkdir data\glove -ErrorAction SilentlyContinue
  Move-Item glove_temp\glove.6B.100d.txt data\glove\
  Remove-Item glove.6B.zip, glove_temp -Recurse -Force
```
- **TextCNN** downloads and caches `glove-wiki-gigaword-100` automatically via `gensim.downloader` the first
  time its preprocessing script runs (no manual step needed, but the first run takes a few minutes).

**DistilBERT** needs neither: its own pretrained tokenizer and weights (`distilbert-base-uncased`) are
downloaded automatically from the Hugging Face Hub the first time its script runs.

## 4. Running each model

Each model follows the same three-step pattern: preprocess/embeddings (if applicable) → train → evaluate.
Run all commands from the repository root.

### MLP

```bash
uv run python -m models.mlp.preprocessing   # builds vocab.json
uv run python -m models.mlp.embeddings      # builds the GloVe embedding matrix
uv run python -m models.mlp.train           # trains, saves best_model.pt + config.json + history.json
uv run python -m models.mlp.evaluate        # evaluates on the test set, saves metrics + plots
uv run python -m models.mlp.compare_experiments  # fine-tuned vs frozen embeddings comparison
```

### BiLSTM

```bash
uv run python -m models.bilstm.preprocessing
uv run python -m models.bilstm.embeddings
uv run python -m models.bilstm.train
uv run python -m models.bilstm.evaluate
uv run python -m models.bilstm.compare_experiments
```

### TextCNN

```bash
uv run python -m models.textcnn.preprocessing
uv run python -m models.textcnn.train
uv run python -m models.textcnn.evaluate
uv run python -m models.textcnn.experiments   # ablation study (frozen/random embeddings, multi-seed, GPU check)
```

### DistilBERT

```bash
uv run python -m models.distilbert.train
uv run python -m models.distilbert.evaluate
uv run python -m models.distilbert.compare_experiments
```

For MLP and BiLSTM, the fine-tuned-vs-frozen experiment is selected by editing `EXPERIMENT_NAME` in `train.py`
and `evaluate.py` (`"fine_tuned"` or `"frozen"`) before running; results are saved separately under
`results/<model>/experiments/<experiment_name>/`.

### Comparing all four models together

Once every model above has been trained and evaluated at least once:

```bash
uv run python -m src.common.compare_all_models
```

This produces a combined table and two charts (`results/comparison/`) comparing accuracy, macro F1, ROC-AUC,
parameter count, model size, training time and inference time across all four models.

## 5. Configuration and reproducibility

- **Random seed:** 42, fixed for the data split and for every model's training run.
- **Hyperparameters:** each model's full configuration (learning rate, batch size, dropout, epochs, etc.) is
  hardcoded near the top of its `train.py` and is also written out to `results/<model>/.../config.json` after
  every run, alongside the resulting metrics.
- **Metrics:** computed identically for every model by `src/common/metrics.py` (accuracy, macro precision/
  recall/F1, macro one-vs-rest ROC-AUC, confusion matrix, top-confusions list).

## 6. Repository structure
- `data/splits/` — shared train/val/test CSVs + `labels.json` (generated by `src/common/data.py`)
- `notebooks/01_eda.ipynb` - exploratory data analysis
- `src/common/` - `data.py` (shared split), `metrics.py` (shared evaluation), `compare_all_models.py` (cross-model comparison)
- `models/`
  - `mlp/`, `bilstm/`, `textcnn/`, `distilbert/` - each contains `preprocessing.py`, `model.py`, `train.py`, `evaluate.py`
- `results/`
  - `eda/` — EDA figures
  - `mlp/`, `bilstm/`, `textcnn/`, `distilbert/` - per-model metrics, plots, confusion matrices
  - `comparison/` - cross-model comparison table + charts
- `pyproject.toml`, `uv.lock`, `requirements.txt`
- `README.md`

Trained model weights (`*.pt`, `*.safetensors`) and the raw GloVe file are not committed (see `.gitignore`) re-running the training commands above regenerates them.

## 7. Results summary

| Model | Test accuracy | Macro F1 | Parameters |
|---|---|---|---|
| MLP | 89.09% | 0.8911 | 168,361 |
| BiLSTM | 88.15% | 0.8820 | 400,809 |
| TextCNN | 91.10% | 0.9110 | 258,977 |
| DistilBERT | 92.18% | 0.9220 | 67,012,685 |

Full results, learning curves, confusion matrices and analysis are in `results/` and in the project report.
