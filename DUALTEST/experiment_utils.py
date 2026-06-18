import os
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_DIR = DATASET_DIR / "metadata"


DATASET_METADATA = {
    "wikimia": "wikimia_metadata.csv",
    "wikimia24": "wikimia24_metadata.csv",
    "wikimia25": "wikimia25_metadata.csv",
    "wikimia_extended": "wikimia_extended_metadata.csv",
    "booktection": "booktection_metadata.csv",
    "news_recent": "articulos_actuales_metadata.csv",
    "news_historical": "articulos_historicos_es_metadata.csv",
}


def load_metadata(dataset_name: str) -> pd.DataFrame:
    """
    Carga el CSV de metadata generado por build_all_datasets.py.
    """
    if dataset_name not in DATASET_METADATA:
        raise ValueError(
            f"Dataset desconocido: {dataset_name}. "
            f"Opciones válidas: {list(DATASET_METADATA.keys())}"
        )

    metadata_path = METADATA_DIR / DATASET_METADATA[dataset_name]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No encontré el archivo de metadata: {metadata_path}\n"
            "Verificá que ya hayas corrido build_all_datasets.py desde TP_NLP."
        )

    df = pd.read_csv(metadata_path)

    if "file_path" not in df.columns:
        raise ValueError(f"El metadata de {dataset_name} no tiene columna file_path.")

    return df


def normalize_file_path(file_path: str) -> Path:
    """
    Convierte rutas relativas tipo dataset/raw/... a rutas absolutas dentro de TP_NLP.
    """
    path = Path(file_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_text(file_path: str) -> str:
    """
    Abre el .txt asociado a una fila de metadata.
    """
    path = normalize_file_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No encontré el texto: {path}")

    return path.read_text(encoding="utf-8").strip()


def filter_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Aplica filtros específicos por dataset.
    Para BookTection, nos quedamos solo con el texto original, no con las opciones distractoras.
    """
    df = df.copy()

    if dataset_name == "booktection":
        if "is_original" not in df.columns:
            raise ValueError("BookTection debería tener columna is_original.")
        df = df[df["is_original"] == True].copy()

    return df.reset_index(drop=True)


def sample_records(
    df: pd.DataFrame,
    n: int | None = None,
    random_state: int = 7,
    balance_labels: bool = False,
) -> pd.DataFrame:
    """
    Devuelve una muestra del dataset.
    Si balance_labels=True, intenta tomar la misma cantidad de miembros y no-miembros.
    """
    df = df.copy()

    if n is None or n >= len(df):
        return df.reset_index(drop=True)

    if balance_labels and "label" in df.columns:
        labels = sorted(df["label"].dropna().unique())

        if len(labels) == 2:
            per_label = n // 2
            parts = []

            for label in labels:
                label_df = df[df["label"] == label]
                take = min(per_label, len(label_df))
                parts.append(label_df.sample(take, random_state=random_state))

            sampled = pd.concat(parts, ignore_index=True)

            if len(sampled) < n:
                remaining = df.drop(sampled.index, errors="ignore")
                extra = remaining.sample(
                    min(n - len(sampled), len(remaining)),
                    random_state=random_state,
                )
                sampled = pd.concat([sampled, extra], ignore_index=True)

            return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df.sample(n, random_state=random_state).reset_index(drop=True)


def prepare_records(
    dataset_name: str,
    n: int | None = None,
    random_state: int = 7,
    balance_labels: bool = False,
) -> list[dict]:
    """
    Carga metadata + abre textos + devuelve registros listos para DUALTEST.
    """
    df = load_metadata(dataset_name)
    df = filter_dataset(df, dataset_name)
    df = sample_records(
        df,
        n=n,
        random_state=random_state,
        balance_labels=balance_labels,
    )

    records = []

    for idx, row in df.iterrows():
        text = load_text(row["file_path"])

        records.append({
            "id": row.get("file_name", f"{dataset_name}_{idx}"),
            "dataset": dataset_name,
            "dataset_family": row.get("dataset_family", dataset_name),
            "source_dataset": row.get("source_dataset", None),
            "file_path": row["file_path"],
            "label": int(row["label"]) if "label" in row and pd.notna(row["label"]) else None,
            "estimated_membership": row.get("estimated_membership", None),
            "text": text,
            "text_hash": row.get("text_hash", None),
        })

    return records


def load_all_dataset_names() -> list[str]:
    return list(DATASET_METADATA.keys())