import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    # import como paquete (uso normal desde la raiz del repo, ej. agents/tools/)
    from DUALTEST.experiment_utils import prepare_records
    from DUALTEST.prefixing import split_by_tokens
    from DUALTEST.target_model import HFLocalTarget
    from DUALTEST.reference_model import ReferenceModel
    from DUALTEST.metrics import rlb_score, esb_score
except ImportError:
    # fallback para notebooks existentes que hacen sys.path.append(".../DUALTEST")
    # y despues `from run_experiment import ...` con imports bare
    from experiment_utils import prepare_records
    from prefixing import split_by_tokens
    from target_model import HFLocalTarget
    from reference_model import ReferenceModel
    from metrics import rlb_score, esb_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment(
    dataset_name: str,
    target_model_name: str,
    reference_model_name: str,
    n: int | None,
    random_state: int,
    balance_labels: bool,
    prefix_len: int,
    continuation_len: int,
    max_new_tokens: int,
    output_name: str | None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Dataset:", dataset_name)
    print("Target:", target_model_name)
    print("Reference:", reference_model_name)

    records = prepare_records(
        dataset_name=dataset_name,
        n=n,
        random_state=random_state,
        balance_labels=balance_labels,
    )

    print("Registros cargados:", len(records))

    target = HFLocalTarget(
        model_name=target_model_name,
        device=device,
    )

    reference = ReferenceModel(
        model_name=reference_model_name,
        device=device,
    )

    results = []

    for record in tqdm(records):
        try:
            split = split_by_tokens(
                record["text"],
                tokenizer=target.tokenizer,
                prefix_len=prefix_len,
                continuation_len=continuation_len,
            )

            completion = target.complete(
                split.prefix_text,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            target_tokens = completion.token_ids
            source_tokens = split.continuation_token_ids
            prefix_tokens = split.prefix_token_ids

            r, p_rlb = rlb_score(
                target_tokens=target_tokens,
                source_tokens=source_tokens,
                reference_model=reference,
                prefix_token_ids=prefix_tokens,
            )

            s, p_esb = esb_score(
                target_text=completion.text,
                target_tokens=target_tokens,
                source_text=split.continuation_text,
                reference_model=reference,
                prefix_token_ids=prefix_tokens,
            )

            results.append({
                "id": record["id"],
                "dataset": record["dataset"],
                "label": record["label"],
                "membership": record["estimated_membership"],
                "run_length": r,
                "p_rlb": p_rlb,
                "edit_similarity": s,
                "p_esb": p_esb,
                "prefix": split.prefix_text,
                "ground_truth": split.continuation_text,
                "target_completion": completion.text,
                "target_model": target_model_name,
                "reference_model": reference_model_name,
            })

        except Exception as e:
            results.append({
                "id": record.get("id"),
                "dataset": dataset_name,
                "label": record.get("label"),
                "membership": record.get("estimated_membership"),
                "error": str(e),
                "target_model": target_model_name,
                "reference_model": reference_model_name,
            })

    df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(exist_ok=True)

    if output_name is None:
        safe_target = target_model_name.replace("/", "_")
        safe_ref = reference_model_name.replace("/", "_")
        output_name = f"dualtest_{dataset_name}_{safe_target}_ref_{safe_ref}_n{len(records)}.csv"

    output_path = RESULTS_DIR / output_name
    df.to_csv(output_path, index=False)

    print("Guardado en:", output_path)

    if "error" in df.columns:
        print("Errores:", df["error"].notna().sum())

    if "run_length" in df.columns:
        print("\nResumen:")
        print(df[["label", "run_length", "edit_similarity", "p_rlb", "p_esb"]].groupby("label").mean(numeric_only=True))

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--reference", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--balance-labels", action="store_true")
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--continuation-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        target_model_name=args.target,
        reference_model_name=args.reference,
        n=args.n,
        random_state=args.random_state,
        balance_labels=args.balance_labels,
        prefix_len=args.prefix_len,
        continuation_len=args.continuation_len,
        max_new_tokens=args.max_new_tokens,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()