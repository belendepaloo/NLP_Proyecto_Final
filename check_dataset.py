import os
import pandas as pd


METADATA_FILES = {
    "wikimia": "dataset/metadata/wikimia_metadata.csv",
    "wikimia24": "dataset/metadata/wikimia24_metadata.csv",
    "wikimia_extended": "dataset/metadata/wikimia_extended_metadata.csv",
    "wikimia25": "dataset/metadata/wikimia25_metadata.csv",
    "booktection": "dataset/metadata/booktection_metadata.csv",
    "articulos_actuales": "dataset/metadata/articulos_actuales_metadata.csv",
    "articulos_historicos_es": "dataset/metadata/articulos_historicos_es_metadata.csv",
}

RECENT_MONTH_QUOTAS = {
    (2025, 10): 500,
    (2025, 11): 500,
    (2025, 12): 500,
    (2026, 1): 250,
    (2026, 2): 250,
    (2026, 3): 250,
    (2026, 4): 250,
}

HISTORICAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def read_existing_metadata():
    data = {}
    for name, path in METADATA_FILES.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
    return data


def check_metadata_exists():
    print("\n1) CHECK METADATA FILES")
    ok = True

    for name, path in METADATA_FILES.items():
        exists = os.path.exists(path)
        print(f"{name}: {'OK' if exists else 'FALTA'} - {path}")
        if not exists:
            ok = False

    return ok


def check_basic_structure(data):
    print("\n2) CHECK BASIC STRUCTURE")
    ok = True
    required_columns = ["file_name", "file_path", "label", "estimated_membership"]

    for name, df in data.items():
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            print(f"{name}: ERROR - faltan columnas {missing}")
            ok = False
        else:
            print(f"{name}: OK")

    return ok


def check_labels(data):
    print("\n3) CHECK LABELS")
    ok = True

    for name, df in data.items():
        labels = set(df["label"].dropna().unique())
        print(f"\n{name}")
        print(df["label"].value_counts(dropna=False))

        if not labels.issubset({0, 1}):
            print("ERROR: labels inválidos:", labels)
            ok = False
        else:
            print("OK")

    return ok


def check_membership_consistency(data):
    print("\n4) CHECK MEMBERSHIP CONSISTENCY")
    ok = True

    for name, df in data.items():
        expected = df["label"].apply(lambda x: "member" if int(x) == 1 else "non_member")
        invalid = df[df["estimated_membership"] != expected]

        print(f"{name}: inconsistencias {len(invalid)}")

        if len(invalid) > 0:
            ok = False

    return ok


def check_file_paths(data):
    print("\n5) CHECK FILE PATHS")
    ok = True

    for name, df in data.items():
        missing = []
        empty = []

        for file_path in df["file_path"]:
            if not os.path.exists(file_path):
                missing.append(file_path)
            elif os.path.getsize(file_path) == 0:
                empty.append(file_path)

        print(f"\n{name}")
        print("archivos esperados:", len(df))
        print("faltantes:", len(missing))
        print("vacíos:", len(empty))

        if missing:
            print("ejemplos faltantes:", missing[:5])
            ok = False

        if empty:
            print("ejemplos vacíos:", empty[:5])
            ok = False

        if not missing and not empty:
            print("OK")

    return ok


def check_duplicates(data):
    print("\n6) CHECK DUPLICATES")
    ok = True

    no_duplicate_expected = [
        "wikimia_extended",
        "articulos_actuales",
        "articulos_historicos_es",
    ]

    for name, df in data.items():
        if "text_hash" not in df.columns:
            print(f"{name}: SKIP")
            continue

        total = len(df)
        unique = df["text_hash"].nunique()
        duplicates = total - unique

        print(f"\n{name}")
        print("total:", total)
        print("hashes únicos:", unique)
        print("duplicados exactos:", duplicates)

        if name in no_duplicate_expected and duplicates != 0:
            print("ERROR: no debería tener duplicados")
            ok = False
        else:
            print("OK")

    return ok


def check_wikimia_extended(data):
    print("\n7) CHECK WIKIMIA EXTENDED")
    if "wikimia_extended" not in data:
        return False

    df = data["wikimia_extended"]
    print(df["source_dataset"].value_counts())

    sources = set(df["source_dataset"].unique())
    ok = {"WikiMIA", "WikiMIA24"}.issubset(sources)

    print("OK" if ok else "ERROR")
    return ok


def check_booktection(data):
    print("\n8) CHECK BOOKTECTION")
    if "booktection" not in data:
        return False

    df = data["booktection"]

    invalid = df[df["is_original"] != (df["option"] == df["answer"])]

    print("filas:", len(df))
    print("lógica incorrecta:", len(invalid))

    ok = len(invalid) == 0
    print("OK" if ok else "ERROR")
    return ok


def check_recent_articles(data):
    print("\n9) CHECK ARTICULOS ACTUALES")
    if "articulos_actuales" not in data:
        return False

    df = data["articulos_actuales"]
    ok = True

    print("cantidad:", len(df))
    print("rango:", df["publish_date"].min(), "->", df["publish_date"].max())

    invalid_dates = df[
        (df["publish_date"] < "2025-10-01") |
        (df["publish_date"] > "2026-04-30")
    ]

    print("fechas fuera de rango:", len(invalid_dates))
    print("labels distintos de 0:", len(df[df["label"] != 0]))

    if len(df) == 0 or len(invalid_dates) > 0 or len(df[df["label"] != 0]) > 0:
        ok = False

    if "text_length" in df.columns:
        short = df[df["text_length"] < 800]
        print("textos cortos:", len(short))
        if len(short) > 0:
            ok = False

    print("\npor año:")
    print(df["year"].value_counts().sort_index())

    print("\npor mes:")
    month_counts = df.groupby(["year", "month"]).size().sort_index()
    print(month_counts)

    for (year, month), quota in RECENT_MONTH_QUOTAS.items():
        count = int(month_counts.get((year, month), 0))
        if count < quota:
            print(f"ERROR: {year}-{month:02d} tiene {count}, esperado {quota}")
            ok = False

    print("OK" if ok else "ERROR")
    return ok


def check_historical_articles(data):
    print("\n10) CHECK ARTICULOS HISTORICOS ES")
    if "articulos_historicos_es" not in data:
        return False

    df = data["articulos_historicos_es"]
    ok = True

    print("cantidad:", len(df))
    print("por año:")
    counts = df["year"].value_counts().sort_index()
    print(counts)

    for year in HISTORICAL_YEARS:
        count = int(counts.get(year, 0))
        if count < 500:
            print(f"ERROR: año {year} tiene solo {count} noticias")
            ok = False

    invalid_labels = df[df["label"] != 0]
    print("labels distintos de 0:", len(invalid_labels))

    if len(invalid_labels) > 0:
        ok = False

    if "text_length" in df.columns:
        short = df[df["text_length"] < 800]
        print("textos cortos:", len(short))
        if len(short) > 0:
            ok = False

    if "language" in df.columns:
        non_spanish = df[df["language"] != "spanish"]
        print("idioma distinto de spanish:", len(non_spanish))
        if len(non_spanish) > 0:
            ok = False

    print("OK" if ok else "ERROR")
    return ok


def print_global_summary(data):
    print("\n11) GLOBAL SUMMARY")

    total = 0
    for name, df in data.items():
        print(f"{name}: {len(df)}")
        total += len(df)

    print("TOTAL TXT:", total)


def print_summary(results):
    print("\n==============================")
    print("RESUMEN FINAL")
    print("==============================")

    all_ok = True

    for name, result in results.items():
        print(f"{name}: {'OK' if result else 'ERROR'}")
        if not result:
            all_ok = False

    print("\nRESULTADO GENERAL:")

    if all_ok:
        print("✅ DATASET VALIDADO: todo parece estar correctamente generado.")
    else:
        print("❌ HAY PROBLEMAS: revisar checks con ERROR.")


def main():
    metadata_ok = check_metadata_exists()
    data = read_existing_metadata()

    results = {
        "metadata_exists": metadata_ok,
        "basic_structure": check_basic_structure(data),
        "labels": check_labels(data),
        "membership_consistency": check_membership_consistency(data),
        "file_paths": check_file_paths(data),
        "duplicates": check_duplicates(data),
        "wikimia_extended": check_wikimia_extended(data),
        "booktection": check_booktection(data),
        "recent_articles": check_recent_articles(data),
        "historical_articles": check_historical_articles(data),
    }

    print_global_summary(data)
    print_summary(results)


if __name__ == "__main__":
    main()