
import pandas as pd

from DUALTEST_NO.text_utils import (
    read_text,
    split_text_for_dualtest
)

from DUALTEST_NO.metrics import (
    run_length_score,
    edit_similarity,
    token_overlap
)

from DUALTEST_NO.hf_model import generate_completion


def run_dualtest_row(
    row,
    target_model,
    reference_model,
    max_total_words=64,
    prefix_ratio=0.5,
    max_new_tokens=64
):

    text = read_text(row["file_path"])

    prefix, continuation = split_text_for_dualtest(
        text=text,
        max_total_words=max_total_words,
        prefix_ratio=prefix_ratio
    )

    if prefix is None:
        return None

    target_generation = generate_completion(
        prefix,
        model_name=target_model,
        max_new_tokens=max_new_tokens
    )

    reference_generation = generate_completion(
        prefix,
        model_name=reference_model,
        max_new_tokens=max_new_tokens
    )

    target_similarity = edit_similarity(
        continuation,
        target_generation
    )

    reference_similarity = edit_similarity(
        continuation,
        reference_generation
    )

    target_overlap = token_overlap(
        continuation,
        target_generation
    )

    reference_overlap = token_overlap(
        continuation,
        reference_generation
    )

    return {

        "file_name": row["file_name"],
        "estimated_membership": row["estimated_membership"],

        "target_similarity": target_similarity,
        "reference_similarity": reference_similarity,

        "target_overlap": target_overlap,
        "reference_overlap": reference_overlap,

        "dual_similarity_score":
            target_similarity - reference_similarity,

        "dual_overlap_score":
            target_overlap - reference_overlap,

        "target_generation": target_generation,
        "reference_generation": reference_generation
    }


def run_dualtest_dataframe(
    df,
    target_model,
    reference_model,
    max_rows=100
):

    results = []

    for idx, row in df.head(max_rows).iterrows():

        print(
            f"{idx+1}/{max_rows}",
            row["file_name"]
        )

        try:

            result = run_dualtest_row(
                row,
                target_model,
                reference_model
            )

            if result:
                results.append(result)

        except Exception as e:

            print(
                "ERROR",
                row["file_name"],
                e
            )

    return pd.DataFrame(results)
