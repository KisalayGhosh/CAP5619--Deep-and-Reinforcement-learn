import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np

# Load dataset
df = pd.read_csv("regression_input.csv")
df = df.dropna(subset=["utterance", "reward"])
df = df[df["utterance"].str.strip().astype(bool)]  

# Split
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["utterance"].tolist(), df["reward"].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FeedbackDataset(train_texts, train_labels)
test_dataset = FeedbackDataset(test_texts, test_labels)

# Regression model
class BertForRewardRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = torch.nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

# Load model
from transformers import BertConfig
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertForRewardRegression.from_pretrained("bert-base-uncased", config=config)


def compute_metrics(pred):
    preds = pred.predictions
    labels = pred.label_ids
    rmse = mean_squared_error(labels, preds, squared=False)
    corr, _ = pearsonr(labels, preds)
    return {
        "rmse": rmse,
        "pearson": corr
    }


training_args = TrainingArguments(
    output_dir="./reward_model_output",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True
)


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
print("\n Final Reward Regression Results:")
print(results)
