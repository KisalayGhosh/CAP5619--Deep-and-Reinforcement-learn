import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# Loading BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Function to get BERT embedding for a word
def get_word_embedding(word):
    tokens = tokenizer(word, return_tensors='pt', add_special_tokens=False)
    with torch.no_grad():
        output = model(**tokens).last_hidden_state
    return output.mean(dim=1).squeeze().numpy()


def load_analogy_dataset(filepath):
    groups = defaultdict(list)
    current_group = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current_group = line[2:].strip()
            elif line:
                words = line.split()
                if len(words) == 4:
                    groups[current_group].append(words)
    return groups

# Compute top-k accuracy for cosine or L2
def compute_topk_accuracy(analogies, k, use_cosine=True):
    # Get the candidate set (b and d words from the group)
    candidates = list(set([pair[1] for pair in analogies] + [pair[3] for pair in analogies]))
    embeddings = {word: get_word_embedding(word) for word in candidates}
    correct = 0
    total = 0

    for a, b, c, d in analogies:
        try:
            # Compute analogy vector: b - a + c
            vec_a = get_word_embedding(a)
            vec_b = embeddings[b]
            vec_c = get_word_embedding(c)
            vec_d_target = embeddings[d]
        except KeyError:
            continue  

        predicted_vector = vec_b - vec_a + vec_c

        # Similarities to candidate d'
        scores = []
        for cand in candidates:
            if cand in [a, b, c]:  
                continue
            cand_vec = embeddings[cand]
            if use_cosine:
                score = cosine_similarity([predicted_vector], [cand_vec])[0][0]
            else:
                score = -euclidean(predicted_vector, cand_vec)
            scores.append((cand, score))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = [cand for cand, _ in scores[:k]]

        if d in topk:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_group(group_name, analogies, k_values):
    print(f"\n===== Group: {group_name} =====")
    print("k\tCosine Acc\tL2 Acc")
    for k in k_values:
        acc_cos = compute_topk_accuracy(analogies, k, use_cosine=True)
        acc_l2 = compute_topk_accuracy(analogies, k, use_cosine=False)
        print(f"{k}\t{acc_cos:.3f}\t\t{acc_l2:.3f}")


if __name__ == "__main__":
    dataset_path = "word-test.v1.txt" 
    groups = load_analogy_dataset(dataset_path)

    
    selected = ["capital-common-countries", "family", "gram1-adjective-to-adverb"]
    k_values = [1, 2, 5, 10, 20]

    for group_name in selected:
        evaluate_group(group_name, groups[group_name], k_values)
