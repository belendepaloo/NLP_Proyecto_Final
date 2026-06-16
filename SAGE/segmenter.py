import re


class DocumentSegmenter:
    """
    Simple first version of the SAGE segmenter.

    It separates text into:
    - structural: titles, URLs, citations, formulas, code-like lines
    - narrative: normal prose to be paraphrased
    """

    def __init__(self):
        pass

    def is_structural_line(self, line: str) -> bool:
        stripped = line.strip()

        if not stripped:
            return True

        if stripped.startswith(("http://", "https://")):
            return True

        if stripped.startswith(("#", "##", "###", "-", "*", "=", ">", "|")):
            return True

        if re.match(r"^\s*(abstract|title|author|references|introduction)\s*[:=]", stripped.lower()):
            return True

        if re.search(r"\[@.*?\]|\{#.*?\}|\\[a-zA-Z]+|\$.*?\$", stripped):
            return True

        if len(stripped.split()) <= 4 and stripped.endswith(":"):
            return True

        return False

    def split(self, text: str) -> list[dict]:
        if text is None:
            return []

        lines = text.splitlines()
        segments = []

        buffer = []
        buffer_type = None

        def flush():
            nonlocal buffer, buffer_type
            if buffer:
                segments.append({
                    "type": buffer_type,
                    "text": "\n".join(buffer).strip()
                })
                buffer = []
                buffer_type = None

        for line in lines:
            line_type = "structural" if self.is_structural_line(line) else "narrative"

            if buffer_type is None:
                buffer_type = line_type

            if line_type != buffer_type:
                flush()
                buffer_type = line_type

            buffer.append(line)

        flush()

        return [s for s in segments if s["text"]]