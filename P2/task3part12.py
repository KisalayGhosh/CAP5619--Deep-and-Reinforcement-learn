import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Load CSV
df = pd.read_csv("absa_restaurants.csv")
df = df[df["sentiment"].isin(["positive", "neutral", "negative"])]  # Remove any "conflict" if any

# Label encoding
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_map)


df["text"] = df.apply(lambda x: f"{x['sentence']} [SEP] {x['aspect']}", axis=1)

# Split train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create datasets
train_dataset = Dataset.from_dict({**train_encodings, "label": train_labels})
test_dataset = Dataset.from_dict({**test_encodings, "label": test_labels})

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# Training config
training_args = TrainingArguments(
    output_dir="./absa_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
results = trainer.evaluate()
print("\n Final ABSA Evaluation:", results)
