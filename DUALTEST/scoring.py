import pandas as pd

from prefixing import split_by_tokens
from metrics import rlb_score, esb_score


def score_texts(
    texts,
    target,
    reference,
    prefix_len=64,
    continuation_len=64,
    max_new_tokens=64,
    label=0,
    dataset_name="dataset",
):
    rows = []

    for i, text in enumerate(texts):

        split = split_by_tokens(
            text,
            tokenizer=target.tokenizer,
            prefix_len=prefix_len,
            continuation_len=continuation_len,
        )

        completion = target.complete(
            split.prefix_text,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        r, p_rlb = rlb_score(
            target_tokens=completion.token_ids,
            source_tokens=split.continuation_token_ids,
            reference_model=reference,
            prefix_token_ids=split.prefix_token_ids,
        )

        s, p_esb = esb_score(
            target_text=completion.text,
            target_tokens=completion.token_ids,
            source_text=split.continuation_text,
            reference_model=reference,
            prefix_token_ids=split.prefix_token_ids,
        )

        rows.append({
            "id": f"{dataset_name}_{i}",
            "label": label,
            "run_length": r,
            "p_rlb": p_rlb,
            "edit_similarity": s,
            "p_esb": p_esb,
            "prefix": split.prefix_text,
            "ground_truth": split.continuation_text,
            "target_completion": completion.text,
        })

    return pd.DataFrame(rows)