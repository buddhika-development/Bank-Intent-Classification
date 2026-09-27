import pandas as pd
import os
import json
from datasets import Dataset

DATA_DIRR= "../../data/splits"

if not os.path.exists(DATA_DIRR):
    raise ValueError("Data directory not found.")

def load_data():
    train_df = pd.read_csv(f"{DATA_DIRR}/train.csv")
    validation_df = pd.read_csv(f"{DATA_DIRR}/val.csv")

    with open(f"{DATA_DIRR}/labels.json", 'r') as f:
        label_names = json.load(f)

    return train_df, validation_df, label_names

def tokenize_dataset(train_df, validation_df, tokenizer, max_length= 64):

    def tokenizer_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    validation_ds = Dataset.from_pandas(validation_df[['text', 'label']])

    train_ds = train_ds.map(tokenizer_fn, batched= True)
    validation_ds = validation_ds.map(tokenizer_fn, batched= True)

    return train_ds, validation_ds
