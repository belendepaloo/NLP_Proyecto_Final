from segmenter import DocumentSegmenter
from paraphraser import Paraphraser
from wordsim import WordSimilarity
from selector import CandidateSelector
# from sps_light import SPSLight
from sps import SPS


class SAGE:
    """
    SAGE pipeline using SPSLight.

    Flow:
    text
    -> segmenter
    -> paraphraser candidates
    -> SPSLight semantic score
    -> WordSim lexical overlap
    -> selector: max(SPS - WordSim)
    """

    def __init__(self, device: str | None = None):
        self.segmenter = DocumentSegmenter()
        self.sps = SPS(device=device)
        self.paraphraser = Paraphraser()
        self.wordsim = WordSimilarity()
        self.selector = CandidateSelector()

        # self.sps = SPSLight() if use_sps_light else None
        

    def semantic_persistence(self, original: str, candidate: str) -> float:
        if self.sps is None:
            if original.strip() == candidate.strip():
                return 1.0
            return 0.8

        return self.sps.score(original, candidate)

    def score_candidate(self, original: str, candidate: str) -> dict:
        sps = self.semantic_persistence(original, candidate)
        wordsim = self.wordsim.score(original, candidate)

        return {
            "text": candidate,
            "sps": sps,
            "wordsim": wordsim,
            "final_score": sps - wordsim,
        }

    def paraphrase_segment(self, text: str) -> dict:
        candidates_text = self.paraphraser.generate_candidates(text, n=3)

        scored_candidates = [
            self.score_candidate(text, candidate)
            for candidate in candidates_text
        ]

        return self.selector.select(scored_candidates)

    def paraphrase(self, text: str) -> dict:
        segments = self.segmenter.split(text)

        output_segments = []
        details = []

        for segment in segments:
            if segment["type"] == "structural":
                output_segments.append(segment["text"])
                details.append({
                    "type": "structural",
                    "original": segment["text"],
                    "selected": segment["text"],
                    "sps": None,
                    "wordsim": None,
                    "final_score": None,
                })
            else:
                best = self.paraphrase_segment(segment["text"])

                output_segments.append(best["text"])
                details.append({
                    "type": "narrative",
                    "original": segment["text"],
                    "selected": best["text"],
                    "sps": best["sps"],
                    "wordsim": best["wordsim"],
                    "final_score": best["final_score"],
                })

        return {
            "original": text,
            "paraphrase": "\n\n".join(output_segments),
            "segments": details,
        }