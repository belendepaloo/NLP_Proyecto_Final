
import pandas as pd
from pathlib import Path
from text_utils import read_text, split_text_for_dualtest
from metrics import (
    run_length_score,
    edit_similarity,
    first_word_match,
    token_overlap,
)
from hf_model import generate_completion


def run_dualtest_row(
    row,
    model_name="Qwen/Qwen2.5-3B",
    max_total_words=64,
    prefix_ratio=0.5,
    max_new_tokens=64
):
    path = Path("..") / row["file_path"]
    text = read_text(path)

    prefix, continuation = split_text_for_dualtest(
        text=text,
        max_total_words=max_total_words,
        prefix_ratio=prefix_ratio
    )

    if prefix is None:
        return None

    generated = generate_completion(
        prefix=prefix,
        model_name=model_name,
        max_new_tokens=max_new_tokens
    )

    return {
        "file_name": row["file_name"],
        "file_path": row["file_path"],
        "book_id": row.get("book_id"),
        "length": row.get("length"),
        "label": row.get("label"),
        "estimated_membership": row.get("estimated_membership"),
        "prefix": prefix,
        "ground_truth": continuation,
        "generated": generated,
        "run_length": run_length_score(continuation, generated),
        "edit_similarity": edit_similarity(continuation, generated),
        "first_word_match": first_word_match(continuation, generated),
        "token_overlap": token_overlap(continuation, generated),
        "prefix_words": len(prefix.split()),
        "continuation_words": len(continuation.split()),
    }


def run_dualtest_dataframe(
    df,
    n=20,
    model_name="Qwen/Qwen2.5-3B",
    max_total_words=64,
    prefix_ratio=0.5,
    max_new_tokens=64
):
    results = []

    for idx, row in df.head(n).iterrows():
        print("Procesando:", idx, row["file_name"])

        try:
            result = run_dualtest_row(
                row=row,
                model_name=model_name,
                max_total_words=max_total_words,
                prefix_ratio=prefix_ratio,
                max_new_tokens=max_new_tokens
            )

            if result is not None:
                results.append(result)

        except Exception as e:
            print("Error:", row["file_name"], e)

    return pd.DataFrame(results)
