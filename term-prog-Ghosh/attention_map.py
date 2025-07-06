import esm
import torch
import matplotlib.pyplot as plt


model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Load BRCA1 sequence
with open("BRCA1.fasta") as f:
    lines = f.readlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

# Prepare input
data = [("BRCA1", sequence)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Get attention maps
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    attentions = results["attentions"]  # shape: [layers, heads, tokens, tokens]

# Average across heads and layers
attn = attentions.mean(dim=(0, 1))[1:-1, 1:-1]  

# Compute mean attention received by each position
mean_attention_received = attn.sum(dim=0).numpy()


plt.figure(figsize=(12, 5))
plt.plot(range(1, len(mean_attention_received) + 1), mean_attention_received)
plt.xlabel("Amino Acid Position")
plt.ylabel("Total Attention Received")
plt.title("ESM-2 Attention Map for BRCA1")
plt.tight_layout()
plt.savefig("esm2_attention_map.png", dpi=300)
plt.show()
