import time
import joblib
import pandas as pd

from groq import Groq, InternalServerError, RateLimitError, APIConnectionError

from prefixing import split_by_words
from metrics import rlb_score, esb_score
from probability_calibration import (
    calibrate_general_csvs,
    apply_membership_probability_model,
)


def train_and_save_membership_calibrator(
    csv_paths,
    model_path,
    threshold_path,
    output_csv=None,
    target_fpr=0.01,
    include_semantic=False,
):
    model, df_calibrated, metrics = calibrate_general_csvs(
        csv_paths=csv_paths,
        output_csv=output_csv,
        include_semantic=include_semantic,
        target_fpr=target_fpr,
    )

    threshold = metrics["threshold_report"]["threshold"]

    joblib.dump(model, model_path)
    joblib.dump(
        {
            "threshold": threshold,
            "metrics": metrics,
            "target_fpr": target_fpr,
        },
        threshold_path,
    )

    return model, threshold, df_calibrated, metrics


def load_membership_calibrator(model_path, threshold_path):
    model = joblib.load(model_path)
    threshold_info = joblib.load(threshold_path)

    return model, threshold_info["threshold"], threshold_info


def make_groq_completion_fn(
    api_key,
    target_model,
    temperature=0,
):
    client = Groq(api_key=api_key)

    def completion_fn(prompt, max_new_tokens=64, max_retries=8):
        wait = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Continue the following text naturally.\n\n{prompt}",
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content

            except (InternalServerError, RateLimitError, APIConnectionError) as e:
                print(f"Error API intento {attempt + 1}/{max_retries}: {e}")
                time.sleep(wait)
                wait = min(wait * 2, 120)

        raise RuntimeError("Groq falló después de todos los reintentos")

    return completion_fn


def dualtest_pipeline_single(
    source_text,
    completion_fn,
    reference,
    membership_model,
    threshold,
    target_model_name,
    reference_model_name="EleutherAI/pythia-410m",
    prefix_words=50,
    continuation_len=64,
    max_new_tokens=64,
):
    split = split_by_words(
        source_text,
        prefix_len=prefix_words,
        continuation_len=continuation_len,
    )

    prefix_ids = reference.tokenizer(
        split.prefix_text,
        add_special_tokens=False,
    ).input_ids

    source_tokens = reference.tokenizer(
        split.continuation_text,
        add_special_tokens=False,
    ).input_ids[:continuation_len]

    target_completion = completion_fn(
        split.prefix_text,
        max_new_tokens=max_new_tokens,
    )

    target_tokens = reference.tokenizer(
        target_completion,
        add_special_tokens=False,
    ).input_ids[:max_new_tokens]

    r, p_rlb = rlb_score(
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        reference_model=reference,
        prefix_token_ids=prefix_ids,
    )

    s, p_esb = esb_score(
        target_text=target_completion,
        target_tokens=target_tokens,
        source_text=split.continuation_text,
        reference_model=reference,
        prefix_token_ids=prefix_ids,
    )

    log_p_esb = reference.sequence_log_probability(
        prefix_ids,
        target_tokens,
    )

    row = pd.DataFrame([{
        "run_length": r,
        "edit_similarity": s,
        "p_rlb": p_rlb,
        "p_esb": p_esb,
        "log_p_esb": log_p_esb,
    }])

    row["neg_log_p_esb"] = -row["log_p_esb"]
    row["esb_evidence"] = row["neg_log_p_esb"] * row["edit_similarity"]

    row = apply_membership_probability_model(
        row,
        membership_model,
        include_semantic=False,
    )

    membership_probability = float(row["membership_probability"].iloc[0])
    suspicious = membership_probability >= threshold

    return {
        "target_model": target_model_name,
        "reference_model": reference_model_name,

        "run_length": int(r),
        "edit_similarity": float(s),
        "p_rlb": float(p_rlb),
        "p_esb": float(p_esb),
        "log_p_esb": float(log_p_esb),

        "membership_probability": membership_probability,
        "membership_threshold": float(threshold),
        "suspicious": bool(suspicious),

        "prefix": split.prefix_text,
        "ground_truth": split.continuation_text,
        "target_completion": target_completion,
    }