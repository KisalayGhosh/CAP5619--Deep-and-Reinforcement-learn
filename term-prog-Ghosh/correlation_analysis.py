# mutant_sensitivity.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from esm_embedder import ESMEmbedder
from dnn_classifier import DNNClassifier


def generate_mutants(sequence: str, position: int):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    mutants = []
    for aa in aa_list:
        if aa != sequence[position]:
            mutated = sequence[:position] + aa + sequence[position+1:]
            mutants.append((aa, mutated))
    return mutants


def run_mutation_scan(model, embedder, original_seq, position, input_dim, device):
    mutants = generate_mutants(original_seq, position)
    results = {}

    for aa, seq in mutants:
        emb = embedder.embed_sequence(seq)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        results[aa] = prob
    return results


def plot_mutation_effects(results, position):
    aa = list(results.keys())
    probs = [results[a] for a in aa]

    plt.figure(figsize=(10, 4))
    plt.bar(aa, probs, color='tomato')
    plt.title(f"Predicted Pathogenicity at Position {position}")
    plt.ylabel("Prediction Probability")
    plt.xlabel("Mutated Amino Acid")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/mutant_sensitivity_pos{position}.png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    df = pd.read_csv("df_tp53_test.csv")
    row = df.iloc[0]  
    original_seq = row["sequence"]
    position = 120  

    embedder = ESMEmbedder()
    model = DNNClassifier(input_dim=320)
    model.load_state_dict(torch.load("models/best_dnn_model.pth", map_location=device))
    model.to(device)
    model.eval()

    results = run_mutation_scan(model, embedder, original_seq, position, 320, device)
    plot_mutation_effects(results, position)
