import pandas as pd
import numpy as np
import os
from esm_embedder import ESMEmbedder

def embed_and_save(csv_file, name_prefix):
    print(f"Processing {csv_file}...")

    df = pd.read_csv(csv_file)
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()

    embedder = ESMEmbedder()
    embeddings = []

    for seq in sequences:
        emb = embedder.embed_sequence(seq)
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    os.makedirs("embeddings", exist_ok=True)
    np.save(f"embeddings/{name_prefix}_embeddings.npy", embeddings)
    np.save(f"embeddings/{name_prefix}_labels.npy", labels)

    print(f"Saved: embeddings/{name_prefix}_embeddings.npy and _labels.npy")


embed_and_save("df_tp53_train.csv", "train")
embed_and_save("df_tp53_eval.csv", "eval")
embed_and_save("df_tp53_test.csv", "test")
