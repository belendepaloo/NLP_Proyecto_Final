from SAGE.segmenter import DocumentSegmenter
from SAGE.paraphraser import Paraphraser
from SAGE.wordsim import WordSimilarity
from SAGE.selector import CandidateSelector
# from sps_light import SPSLight
from SAGE.sps import SPS


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

    def __init__(self, device: str | None = None, min_length_ratio: float = 0.75):
        self.segmenter = DocumentSegmenter()
        self.sps = SPS(device=device)
        self.paraphraser = Paraphraser()
        self.wordsim = WordSimilarity()
        self.selector = CandidateSelector()

        # Largo mínimo (como fracción del original, en caracteres) que debe
        # tener un candidato para ser considerado válido. Filtra paráfrasis
        # truncadas/resumidas antes de hacerles SPS+WordSim.
        self.min_length_ratio = min_length_ratio

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
    

    # def paraphrase_segment(self, text: str) -> dict:
    #     candidates_text = self.paraphraser.generate_candidates(text, n=3)

    #     scored_candidates = [
    #         self.score_candidate(text, candidate)
    #         for candidate in candidates_text
    #     ]

    #     return self.selector.select(scored_candidates)


    def paraphrase_segment(self, text: str) -> dict:
        print("\nSEGMENT LENGTH:", len(text))
        print("SEGMENT:")
        print(text)
        print("-" * 80)
        candidates_text = self.paraphraser.generate_candidates(text, n=3, min_length_ratio=self.min_length_ratio)

        scored_candidates = [self.score_candidate(text, candidate) for candidate in candidates_text]

        best = self.selector.select(scored_candidates)
        best["all_candidates"] = scored_candidates
        return best
    

    def paraphrase(self, text: str) -> dict:
        segments = self.segmenter.split(text)

        # FAST PATH:
        # textos cortos que producen un único segmento narrativo
        if (len(segments) == 1 and segments[0]["type"] == "narrative"):
            best = self.paraphrase_segment(segments[0]["text"])

            return {
                "original": text,
                "paraphrase": best["text"],
                "segments": [{
                    "type": "narrative",
                    "original": segments[0]["text"],
                    "selected": best["text"],
                    "sps": best["sps"],
                    "wordsim": best["wordsim"],
                    "final_score": best["final_score"],
                    "all_candidates": best["all_candidates"],
                }],
            }

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
                    "all_candidates": best["all_candidates"],
                })

        return {
            "original": text,
            "paraphrase": "\n\n".join(output_segments),
            "segments": details,
        }