from SAGE.metrics import OverlapMetrics


class WordSimilarity:
    """
    WordSim = (token_jaccard + token_trigram_overlap + char_5gram_overlap) / 3

    Lower WordSim means the candidate is more lexically different
    from the original text.
    """

    def __init__(self):
        pass

    def score(self, original: str, candidate: str) -> float:
        return OverlapMetrics.combined(original, candidate)

    def detailed_score(self, original: str, candidate: str) -> dict:
        return {
            "token_jaccard": OverlapMetrics.token_jaccard(original, candidate),
            "token_trigram_overlap": OverlapMetrics.token_trigram_overlap(original, candidate),
            "char_5gram_overlap": OverlapMetrics.char_5gram_overlap(original, candidate),
            "wordsim": OverlapMetrics.combined(original, candidate),
        }