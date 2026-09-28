"""
Fine-tune DistilBERT on BANKING77.

Run:
    uv run python -m models.distilbert.train
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from transformers import (
    DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
)

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
sys.path.append(str(ROOT))

from src.common.data import load_splits
from src.common.metrics import compute_metrics
from models.distilbert.preprocessing import MODEL_NAME, MAX_LEN, load_tokenizer, tokenize_dataset
from models.distilbert.model import build_model

# ---- Config: everything that defines this run, so it's reproducible ----
EXPERIMENT_NAME = "fine_tuned"  # change this per experiment: "fine_tuned", "frozen"

SEED = 42
FREEZE_ENCODER = EXPERIMENT_NAME == "frozen"
BATCH_SIZE = 32
# Only the small head trains when frozen, so it needs a much larger learning rate and more epochs
LEARNING_RATE = 1e-3 if FREEZE_ENCODER else 3e-5
MAX_EPOCHS = 30 if FREEZE_ENCODER else 10
PATIENCE = 4          # stop if val loss doesn't improve for this many epochs
WEIGHT_DECAY = 0.01

RESULTS_DIR = ROOT / "results" / "distilbert" / "experiments" / EXPERIMENT_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def trainer_metrics(eval_pred):
    """Adapter so the Trainer reports the same shared metrics as every other model."""
    logits, labels = eval_pred
    return compute_metrics(labels, np.argmax(logits, axis=-1))


def extract_history(log_history: list[dict]) -> dict:
    """Rebuild the same per-epoch history format the MLP/BiLSTM training loops write.
    train_loss is the running loss during the epoch (as in MLP/BiLSTM); train_acc comes
    from an eval pass over the train set at the end of the epoch, since Trainer doesn't track it."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for log in log_history:
        if "loss" in log:
            history["train_loss"].append(log["loss"])
        if "eval_train_accuracy" in log:
            history["train_acc"].append(log["eval_train_accuracy"])
        if "eval_val_loss" in log:
            history["val_loss"].append(log["eval_val_loss"])
            history["val_acc"].append(log["eval_val_accuracy"])
    return history


def main():
    set_seed(SEED)  # before build_model, so the randomly initialised head is reproducible

    train_df, val_df, test_df, label_names = load_splits()

    tokenizer = load_tokenizer()
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)

    model = build_model(label_names, freeze_encoder=FREEZE_ENCODER)

    experiment_dir = MODEL_DIR / "experiments" / EXPERIMENT_NAME
    checkpoint_dir = experiment_dir / "checkpoints"
    best_model_path = experiment_dir / "best_model"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="val_loss",
        save_total_limit=1,
        save_only_model=True,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_ds,
        eval_dataset={"train": train_ds, "val": val_ds},
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.1f} seconds ({training_time/60:.1f} min)")

    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    history = extract_history(trainer.state.log_history)
    config = {
        "seed": SEED, "model_name": MODEL_NAME, "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "freeze_encoder": FREEZE_ENCODER,
        "epochs_run": len(history["val_loss"]), "training_time_seconds": round(training_time, 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_loss": trainer.state.best_metric,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved: {best_model_path}")
    print(f"Saved: {RESULTS_DIR / 'config.json'}")
    print(f"Saved: {RESULTS_DIR / 'history.json'}")


if __name__ == "__main__":
    main()
