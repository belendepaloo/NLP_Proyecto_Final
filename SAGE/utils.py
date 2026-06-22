import os
import pandas as pd

PARAPHRASER_NAME = os.environ.get("SAGE_PARAPHRASER", "t5")

OUTPUT_DIR = f"dataset/sage_outputs/{PARAPHRASER_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_checkpoint_path(dataset_name: str) -> str:
    return os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_checkpoint_{PARAPHRASER_NAME}.csv"
    )


def load_checkpoint(dataset_name: str) -> tuple[list, set]:
    checkpoint_path = get_checkpoint_path(dataset_name)

    if not os.path.exists(checkpoint_path):
        return [], set()

    print(f"Retomando desde checkpoint: {checkpoint_path}")

    df_checkpoint = pd.read_csv(checkpoint_path)

    results = df_checkpoint.to_dict("records")
    already_processed = set(df_checkpoint["file_name"].tolist())

    print(f"Samples ya procesados: {len(already_processed)}")

    return results, already_processed


def save_checkpoint(results: list, dataset_name: str) -> None:
    checkpoint_path = get_checkpoint_path(dataset_name)
    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
    print(f"Checkpoint actualizado: {checkpoint_path}")


def cleanup_checkpoints(dataset_name: str) -> None:
    checkpoint_path = get_checkpoint_path(dataset_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint borrado: {checkpoint_path}")