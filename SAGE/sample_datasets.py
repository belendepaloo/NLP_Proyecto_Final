import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
METADATA_DIR = os.path.join(BASE, "dataset/metadata")

N_SAMPLES = 200
RANDOM_STATE = 42

DATASETS = [
    {
        "name": "wikimia",
        "input": "wikimia_metadata.csv",
        "filter": lambda df: df[df["file_name"].str.extract(r"length(\d+)")[0].astype(int) == 128],
    },
    {
        "name": "wikimia24",
        "input": "wikimia24_metadata.csv",
        "filter": lambda df: df[df["file_name"].str.extract(r"length(\d+)")[0].astype(int) == 128],
    },
    {
        "name": "booktection",
        "input": "booktection_metadata.csv",
        "filter": lambda df: df[df["option"] == "A"],
    },
]

for ds in DATASETS:
    input_path = os.path.join(METADATA_DIR, ds["input"])
    output_path = os.path.join(METADATA_DIR, f"{ds['name']}_sampled_SAGE.csv")

    print(f"\n=== {ds['name'].upper()} ===")

    if not os.path.exists(input_path):
        print(f"SKIP — no se encontró: {input_path}")
        continue

    df = pd.read_csv(input_path)
    print(f"Total en metadata: {len(df)}")

    df_filtered = ds["filter"](df)
    print(f"Tras filtro: {len(df_filtered)}")

    if len(df_filtered) < N_SAMPLES:
        print(f"ADVERTENCIA: solo hay {len(df_filtered)} textos, se usan todos.")
        df_sampled = df_filtered.reset_index(drop=True)
    else:
        df_sampled = df_filtered.sample(n=N_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

    df_sampled.to_csv(output_path, index=False)
    print(f"Sampleados: {len(df_sampled)}")
    print(f"Guardado en: {output_path}")

print("\n✓ Listo")