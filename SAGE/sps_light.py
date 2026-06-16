import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SPSLight:
    """
    Lightweight Semantic Persistence Score using HF transformers directly.
    Avoids sentence-transformers / torchcodec issues.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        max_tokens: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_tokens = max_tokens

        print(f"[SPSLight] Loading {model_name} on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        print("[SPSLight] Model loaded.")

    def encode(self, text: str) -> torch.Tensor:
        if text is None or not text.strip():
            return torch.zeros(384)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)

            embedding = summed / counts
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0).detach().cpu().float()

    def score(self, original: str, candidate: str) -> float:
        if not original or not candidate:
            return 0.0

        f1 = self.encode(original)
        f2 = self.encode(candidate)

        return float(F.cosine_similarity(
            f1.unsqueeze(0),
            f2.unsqueeze(0),
            dim=-1
        ).item())