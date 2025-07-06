import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset


df = pd.read_csv("amazon_reviews.csv")


df = df[['reviewText', 'overall']].dropna()
df['label'] = df['overall'].astype(int) - 1  # Ratings 1–5 Labels 0–4

# Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['reviewText'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize reviews
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({**train_encodings, 'label': train_labels})
test_dataset = Dataset.from_dict({**test_encodings, 'label': test_labels})

# Load BERT for classification (5 labels)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Accuracy metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10
)


# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
eval_result = trainer.evaluate()
print(f"\n Final Test Accuracy: {eval_result['eval_accuracy']:.4f}")
