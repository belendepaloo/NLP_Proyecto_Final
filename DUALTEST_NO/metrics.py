
from difflib import SequenceMatcher


def normalize_token(token):
    return token.lower().strip(".,;:!?\"'()[]{}")


def run_length_score(reference, generated):
    ref_tokens = reference.split()
    gen_tokens = generated.split()

    count = 0

    for ref, gen in zip(ref_tokens, gen_tokens):
        if normalize_token(ref) == normalize_token(gen):
            count += 1
        else:
            break

    return count


def edit_similarity(reference, generated):
    return SequenceMatcher(
        None,
        reference.lower(),
        generated.lower()
    ).ratio()


def first_word_match(reference, generated):
    ref_tokens = reference.split()
    gen_tokens = generated.split()

    if len(ref_tokens) == 0 or len(gen_tokens) == 0:
        return 0

    return int(normalize_token(ref_tokens[0]) == normalize_token(gen_tokens[0]))


def token_overlap(reference, generated):
    ref_tokens = set(normalize_token(t) for t in reference.split())
    gen_tokens = set(normalize_token(t) for t in generated.split())

    ref_tokens = set(t for t in ref_tokens if t)
    gen_tokens = set(t for t in gen_tokens if t)

    if len(ref_tokens) == 0:
        return 0

    return len(ref_tokens.intersection(gen_tokens)) / len(ref_tokens)
