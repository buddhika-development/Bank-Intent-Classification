from common.metrics import compute_metrics
import numpy as np
from models.distilbert.data_preprocessing import load_data, tokenize_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

def compute_metric(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

OUTPUT_DIR = "../../results/distilbert"

def train_model(
        model_name= "distilbert-base-uncased",
        epochs= 5,
        batch_size= 32,
        learning_rate= 3e-5
) :
    train_df, validation_df, labels = load_data()
    label_length = len(labels)

    id_to_label = { i : label for i, label in enumerate(labels) }
    label_to_id = { label : i for i, label in enumerate(labels) }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=label_length,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_ds, validation_ds = tokenize_dataset(train_df, validation_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer= tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50
    )

    trainer = Trainer(
        model= model,
        args=training_args,
        data_collator= data_collator,
        train_dataset= train_ds,
        eval_dataset= validation_ds,
        compute_metrics= compute_metric
    )

    print("training process started")
    trainer.train()

    final_metrics = trainer.evaluate()
    print("\nFinal validation metrics:", final_metrics)

    # Save the fine-tuned model + tokenizer together as a local artifact...
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"\nModel saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train_model()