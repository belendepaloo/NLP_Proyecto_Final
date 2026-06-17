import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Paraphraser:
    def __init__(
        self,
        provider: str = "local",
        model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
        device: str | None = None,
    ):
        self.provider = provider

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if provider == "local":
            print(f"[Paraphraser] Loading {model_name} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self.model.eval()
            print("[Paraphraser] Model loaded.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_candidates(self, text: str, n: int = 3) -> list[str]:
        if text is None or not text.strip():
            return []

        if self.provider == "local":
            return self._generate_local(text, n=n)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_local(self, text: str, n: int = 3) -> list[str]:
        prompt = f"paraphrase: {text.strip()}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=max(6, n * 2),
                num_return_sequences=n,
                temperature=1.2,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        candidates = []

        for output in outputs:
            candidate = self.tokenizer.decode(
                output,
                skip_special_tokens=True
            ).strip()

            if candidate and candidate not in candidates:
                candidates.append(candidate)

        return candidates[:n]