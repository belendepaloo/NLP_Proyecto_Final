import re


class OverlapMetrics:
    @staticmethod
    def tokenize(text: str) -> list[str]:
        if text is None:
            return []
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    @staticmethod
    def char_ngrams(text: str, n: int = 5) -> set[str]:
        if text is None:
            return set()

        text = re.sub(r"\s+", " ", text.lower().strip())

        if len(text) < n:
            return {text} if text else set()

        return {text[i:i+n] for i in range(len(text) - n + 1)}

    @staticmethod
    def word_ngrams(tokens: list[str], n: int = 3) -> set[tuple[str, ...]]:
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    @staticmethod
    def token_jaccard(original: str, candidate: str) -> float:
        a = set(OverlapMetrics.tokenize(original))
        b = set(OverlapMetrics.tokenize(candidate))

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        return len(a & b) / len(a | b)

    @staticmethod
    def token_trigram_overlap(original: str, candidate: str) -> float:
        original_tokens = OverlapMetrics.tokenize(original)
        candidate_tokens = OverlapMetrics.tokenize(candidate)

        original_trigrams = OverlapMetrics.word_ngrams(original_tokens, n=3)
        candidate_trigrams = OverlapMetrics.word_ngrams(candidate_tokens, n=3)

        if not original_trigrams:
            return 0.0

        return len(original_trigrams & candidate_trigrams) / len(original_trigrams)

    @staticmethod
    def char_5gram_overlap(original: str, candidate: str) -> float:
        original_grams = OverlapMetrics.char_ngrams(original, n=5)
        candidate_grams = OverlapMetrics.char_ngrams(candidate, n=5)

        if not original_grams:
            return 0.0

        return len(original_grams & candidate_grams) / len(original_grams)

    @staticmethod
    def combined(original: str, candidate: str) -> float:
        jaccard = OverlapMetrics.token_jaccard(original, candidate)
        trigram = OverlapMetrics.token_trigram_overlap(original, candidate)
        char5 = OverlapMetrics.char_5gram_overlap(original, candidate)

        return (jaccard + trigram + char5) / 3