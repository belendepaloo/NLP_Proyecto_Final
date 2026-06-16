class CandidateSelector:
    """
    Selects the best paraphrase candidate using:

        score = SPS - WordSim

    Higher SPS is better.
    Lower WordSim is better.
    """

    def __init__(self):
        pass

    def select(self, candidates: list[dict]) -> dict:
        if not candidates:
            raise ValueError("No candidates provided.")

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:
            sps = candidate.get("sps", 0.0)
            wordsim = candidate.get("wordsim", 1.0)

            final_score = sps - wordsim
            candidate["final_score"] = final_score

            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate

        return best_candidate