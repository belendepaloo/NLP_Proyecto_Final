import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

class SPS:
    """
    Semantic Persistence Score (SPS)
    Paper-faithful implementation:
        - Gemma-2B
        - SAE release: gemma-2b-res-jb
        - Hook point: blocks.12.hook_resid_post
    SPS = cosine_similarity(SAE(original), SAE(candidate))
    """
    def __init__ (self, model_name: str = "gemma-2b", sae_release: str = "gemma-2b-res-jb",
        sae_id: str = "blocks.12.hook_resid_post", device: str | None = None):

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.hook_point = sae_id

        print(f"[SPS] Loading Gemma model on {device}...")

        print("[SPS] Loading Gemma...")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)

        print("[SPS] Loading SAE...")
        sae, _, _= SAE.from_pretrained(release=sae_release,  sae_id=sae_id,  device=device)
        print("[SPS] SAE loaded!")
        self.sae = sae[0] if isinstance(sae, tuple) else sae

    def score(self, original: str, candidate: str) -> float:

        if not original or not candidate:
            return 0.0
        try:
            f1 = self.get_sae_features(original)
            f2 = self.get_sae_features(candidate)
            score = torch.nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0))
            return float(score.item())
        except Exception as e:
            print(f"[SPS] Error computing score: {e}")
            return 0.0
        
    def get_sae_features(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.model.to_tokens(
                text,
                prepend_bos=True,
            )
            _, cache = self.model.run_with_cache(tokens)
            hidden = cache[self.hook_point]
            if hidden.dim() == 3:
                hidden = hidden.mean(dim=1)
            features = self.sae.encode(hidden)
            if features.dim() > 1:
                features = features.squeeze()
            return features.cpu().float()
        
    def encode(self, text: str) -> torch.Tensor:
        return self.get_sae_features(text)


if __name__ == "__main__":

    print("Creating SPS...")
    sps = SPS()
    print("Model and SAE loaded!")
    original = "The cat is sitting on the mat."
    candidate = "The feline is resting on the rug."
    print("Computing score...")
    score = sps.score(original, candidate)
    print(f"SPS = {score:.4f}")