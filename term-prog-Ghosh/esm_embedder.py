from transformers import AutoTokenizer, AutoModel
import torch

class ESMEmbedder:
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_sequence(self, sequence):
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # CLS token
