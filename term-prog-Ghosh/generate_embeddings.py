import pandas as pd
from Bio import SeqIO
import esm
import torch
from tqdm import tqdm

# Load ESM model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Load reference BRCA1 sequence
brca_seq = str(SeqIO.read("BRCA1.fasta", "fasta").seq)

# Load mutants
df = pd.read_csv("mutants.csv")

features = []
positions = []

print(f"Generating embeddings for {len(df)} variants...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    pos, orig, new = int(row["Position"]), row["Original"], row["Mutant"]
    if brca_seq[pos - 1] != orig:
        continue  

    # Create mutated sequence
    mutated_seq = brca_seq[:pos - 1] + new + brca_seq[pos:]

   
    data = [("mut", mutated_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]

    # Extract embedding at the mutation site
    embedding = token_representations[0, pos]  # token index = position (1-based)
    features.append(embedding.numpy())
    positions.append(pos)


import numpy as np
np.savez("mutant_embeddings.npz", features=np.array(features), positions=np.array(positions))
print("Saved ESM-2 embeddings to mutant_embeddings.npz")
