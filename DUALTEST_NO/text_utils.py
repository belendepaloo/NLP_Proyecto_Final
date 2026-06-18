
def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def split_text_for_dualtest(
    text,
    max_total_words=None,
    min_words=20,
    prefix_ratio=0.5
):
    words = text.split()

    if max_total_words is not None:
        words = words[:max_total_words]

    if len(words) < min_words:
        return None, None

    split_idx = int(len(words) * prefix_ratio)

    prefix = " ".join(words[:split_idx])
    continuation = " ".join(words[split_idx:])

    return prefix, continuation
