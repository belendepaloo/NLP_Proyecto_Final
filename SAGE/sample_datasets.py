import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
METADATA_DIR = os.path.join(BASE, "dataset/metadata")

N_SAMPLES = 200
RANDOM_STATE = 42
N_PER_CLASS = N_SAMPLES // 2


DATASETS = [
    {
        "name": "wikimia",
        "input": "wikimia_metadata.csv",
        "filter": lambda df: df[
            df["file_name"].str.extract(r"length(\d+)")[0].astype(int) == 128
        ],
    },
    {
        "name": "wikimia24",
        "input": "wikimia24_metadata.csv",
        "filter": lambda df: df[
            df["file_name"].str.extract(r"length(\d+)")[0].astype(int) == 128
        ],
    },
    {
        "name": "booktection",
        "input": "booktection_metadata.csv",
        "filter": lambda df: df[
            (df["option"] == "A") &
            (df["length"] == "medium")
        ],
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

    members = df_filtered[df_filtered["estimated_membership"] == "member"]
    non_members = df_filtered[df_filtered["estimated_membership"] == "non_member"]

    print(f"Members disponibles: {len(members)}")
    print(f"No-members disponibles: {len(non_members)}")

    if len(members) < N_PER_CLASS or len(non_members) < N_PER_CLASS:

        print("ADVERTENCIA: no hay suficientes muestras balanceadas.")
        n = min(len(members), len(non_members))

        if n == 0:
            print("No se puede balancear, usando muestra normal.")
            df_sampled = df_filtered.sample(
                n=min(N_SAMPLES, len(df_filtered)),
                random_state=RANDOM_STATE
            )

        else:
            members_sample = members.sample(n=n, random_state=RANDOM_STATE)
            non_members_sample = non_members.sample(n=n, random_state=RANDOM_STATE)
            df_sampled = pd.concat([members_sample, non_members_sample])


    else:

        members_sample = members.sample(
            n=N_PER_CLASS,
            random_state=RANDOM_STATE
        )

        non_members_sample = non_members.sample(
            n=N_PER_CLASS,
            random_state=RANDOM_STATE
        )

        df_sampled = pd.concat(
            [members_sample, non_members_sample]
        )


    # Mezclar filas
    df_sampled = df_sampled.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


    # Verificación
    print("\nDistribución final:")
    print(df_sampled["estimated_membership"].value_counts())


    df_sampled.to_csv(output_path, index=False)

    print(f"Sampleados: {len(df_sampled)}")
    print(f"Guardado en: {output_path}")


print("\n✓ Listo")