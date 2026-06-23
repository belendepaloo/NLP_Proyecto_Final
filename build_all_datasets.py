from datasets import load_dataset
import pandas as pd
import os
import hashlib


BASE_DIR = "dataset"

MIN_NEWS_TEXT_LENGTH = 800
HISTORICAL_NEWS_PER_YEAR = 500

RECENT_NEWS_MONTH_QUOTAS = {
    (2025, 10): 500,
    (2025, 11): 500,
    (2025, 12): 500,
    (2026, 1): 250,
    (2026, 2): 250,
    (2026, 3): 250,
    (2026, 4): 250,
}

HISTORICAL_NEWS_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def make_dirs():
    folders = [
        f"{BASE_DIR}/raw/wikimia",
        f"{BASE_DIR}/raw/wikimia24",
        f"{BASE_DIR}/raw/wikimia_extended",
        f"{BASE_DIR}/raw/wikimia25",
        f"{BASE_DIR}/raw/booktection",
        f"{BASE_DIR}/raw/articulos_actuales",
        f"{BASE_DIR}/raw/articulos_historicos_es",
        f"{BASE_DIR}/booktection_complete",
        f"{BASE_DIR}/metadata",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def clean_text(text):
    return "" if text is None else str(text).strip()


def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def looks_spanish(text):
    text = f" {clean_text(text).lower()} "
    words = [
        " el ", " la ", " los ", " las ", " que ", " para ",
        " una ", " por ", " con ", " del ", " en ", " se "
    ]
    return sum(w in text for w in words) >= 5


def is_spanish(row):
    fields = [
        row.get("language_short"),
        row.get("language_iso639_3"),
        row.get("lang"),
    ]
    values = [str(x).lower() for x in fields if x is not None]

    if "spa" in values or "es" in values or "spanish" in values:
        return True

    return looks_spanish(row.get("text", ""))


def export_wikimia_like(hf_name, source_dataset, raw_folder):
    print(f"\nDescargando {source_dataset}...")
    dataset = load_dataset(hf_name)
    rows = []

    for split_name, split_data in dataset.items():
        for i, row in enumerate(split_data):
            text = clean_text(row["input"])
            label = int(row["label"])

            file_name = f"{source_dataset}_{split_name}_{i:05d}.txt"
            file_path = f"{BASE_DIR}/raw/{raw_folder}/{file_name}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            rows.append({
                "file_name": file_name,
                "file_path": file_path,
                "dataset_family": "wikimia",
                "source_dataset": source_dataset,
                "split": split_name,
                "language": "english",
                "label": label,
                "estimated_membership": "member" if label == 1 else "non_member",
                "text_hash": text_hash(text),
                "text": text,
            })

    df = pd.DataFrame(rows)
    df.drop(columns=["text"]).to_csv(
        f"{BASE_DIR}/metadata/{source_dataset.lower()}_metadata.csv",
        index=False
    )

    print(f"{source_dataset} exportado:", len(df))
    print(df["estimated_membership"].value_counts())

    return df


def build_wikimia_extended(wikimia_df, wikimia24_df):
    print("\nConstruyendo WikiMIA Extended...")
    combined = pd.concat([wikimia_df, wikimia24_df], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_hash"], keep="first").reset_index(drop=True)
    print("Duplicados eliminados:", before - len(combined))

    rows = []

    for i, row in combined.iterrows():
        file_name = f"wikimia_extended_{i:05d}.txt"
        file_path = f"{BASE_DIR}/raw/wikimia_extended/{file_name}"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(row["text"])

        rows.append({
            "file_name": file_name,
            "file_path": file_path,
            "dataset_family": "wikimia_extended",
            "source_dataset": row["source_dataset"],
            "original_file_name": row["file_name"],
            "original_split": row["split"],
            "language": "english",
            "label": row["label"],
            "estimated_membership": row["estimated_membership"],
            "text_hash": row["text_hash"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{BASE_DIR}/metadata/wikimia_extended_metadata.csv", index=False)

    print("WikiMIA Extended exportado:", len(df))
    print(df["estimated_membership"].value_counts())


def export_booktection():
    print("\nDescargando BookTection...")
    dataset = load_dataset("avduarte333/BookTection")
    metadata = []

    for i, row in enumerate(dataset["train"]):
        answer = row["Answer"]
        label = int(row["Label"])

        for option in ["A", "B", "C", "D"]:
            text = clean_text(row[f"Example_{option}"])
            file_name = f"booktection_{i:05d}_{option}.txt"
            file_path = f"{BASE_DIR}/raw/booktection/{file_name}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            metadata.append({
                "file_name": file_name,
                "file_path": file_path,
                "dataset_family": "booktection",
                "source_dataset": "BookTection",
                "book_id": row["ID"],
                "option": option,
                "is_original": option == answer,
                "answer": answer,
                "length": row["Length"],
                "language": "english",
                "label": label,
                "estimated_membership": "member" if label == 1 else "non_member",
                "text_hash": text_hash(text),
            })

    df = pd.DataFrame(metadata)
    df.to_csv(f"{BASE_DIR}/metadata/booktection_metadata.csv", index=False)

    dataset["train"].to_pandas().to_csv(
        f"{BASE_DIR}/booktection_complete/booktection_complete.csv",
        index=False
    )

    print("BookTection exportado:", len(df))
    print(df["estimated_membership"].value_counts())


def valid_recent_date(date):
    if date is None:
        return False
    date = str(date)
    return "2025-10-01" <= date <= "2026-04-30"


def valid_year_date(date, year):
    if date is None:
        return False
    date = str(date)
    return f"{year}-01-01" <= date <= f"{year}-12-31"


def iter_infini_month(year, month):
    ds = load_dataset(
        "ruggsea/infini-news-corpus",
        data_files=f"data/year={year}/month={month:02d}/part-*.parquet",
        split="train",
        streaming=True,
    )

    if ds.features is not None and "text_xxhash64" in list(ds.features.keys()):
        ds = ds.remove_columns(["text_xxhash64"])

    return ds


def export_recent_spanish_news():
    print("\nDescargando noticias actuales ES con cupos por mes...")
    out_dir = f"{BASE_DIR}/raw/articulos_actuales"

    metadata = []
    seen_hashes = set()

    for (year, month), quota in RECENT_NEWS_MONTH_QUOTAS.items():
        print(f"\nProcesando recientes {year}-{month:02d}, objetivo={quota}")
        collected = 0

        for row in iter_infini_month(year, month):
            if collected >= quota:
                break

            text = clean_text(row.get("text"))
            h = text_hash(text)

            if len(text) < MIN_NEWS_TEXT_LENGTH:
                continue
            if not valid_recent_date(row.get("publish_date")):
                continue
            if not is_spanish(row):
                continue
            if h in seen_hashes:
                continue

            seen_hashes.add(h)

            file_name = f"articulo_actual_{len(metadata):05d}.txt"
            file_path = f"{out_dir}/{file_name}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            metadata.append({
                "file_name": file_name,
                "file_path": file_path,
                "dataset_family": "articulos_actuales",
                "source_dataset": "INFINI-NEWS",
                "temporal_bucket": f"{year}-{month:02d}",
                "title": row.get("title"),
                "url": row.get("url"),
                "publish_date": row.get("publish_date"),
                "year": year,
                "month": month,
                "language": "spanish",
                "language_short": row.get("language_short"),
                "language_iso639_3": row.get("language_iso639_3"),
                "lang": row.get("lang"),
                "iptc_topic": row.get("iptc_topic"),
                "text_length": len(text),
                "label": 0,
                "estimated_membership": "non_member",
                "text_hash": h,
            })

            collected += 1

        print(f"Recolectadas {year}-{month:02d}: {collected}")

        if collected < quota:
            print(f"ADVERTENCIA: no se llegó al cupo de {quota} para {year}-{month:02d}")

    df = pd.DataFrame(metadata)
    df.to_csv(f"{BASE_DIR}/metadata/articulos_actuales_metadata.csv", index=False)

    print("\nNoticias actuales exportadas:", len(df))
    if len(df) > 0:
        print("Rango:", df["publish_date"].min(), "->", df["publish_date"].max())
        print(df.groupby(["year", "month"]).size())


def export_historical_spanish_news():
    print("\nDescargando noticias históricas ES...")
    out_dir = f"{BASE_DIR}/raw/articulos_historicos_es"

    metadata = []
    global_seen_hashes = set()

    for year in HISTORICAL_NEWS_YEARS:
        print(f"\nProcesando año histórico {year}...")
        year_count = 0

        for month in range(1, 13):
            if year_count >= HISTORICAL_NEWS_PER_YEAR:
                break

            print(f"  mes {month:02d}...")

            for row in iter_infini_month(year, month):
                if year_count >= HISTORICAL_NEWS_PER_YEAR:
                    break

                text = clean_text(row.get("text"))
                h = text_hash(text)

                if len(text) < MIN_NEWS_TEXT_LENGTH:
                    continue
                if not valid_year_date(row.get("publish_date"), year):
                    continue
                if not is_spanish(row):
                    continue
                if h in global_seen_hashes:
                    continue

                global_seen_hashes.add(h)

                file_name = f"articulo_historico_es_{year}_{year_count:04d}.txt"
                file_path = f"{out_dir}/{file_name}"

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

                metadata.append({
                    "file_name": file_name,
                    "file_path": file_path,
                    "dataset_family": "articulos_historicos_es",
                    "source_dataset": "INFINI-NEWS",
                    "temporal_bucket": str(year),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "publish_date": row.get("publish_date"),
                    "year": year,
                    "month": month,
                    "language": "spanish",
                    "language_short": row.get("language_short"),
                    "language_iso639_3": row.get("language_iso639_3"),
                    "lang": row.get("lang"),
                    "iptc_topic": row.get("iptc_topic"),
                    "text_length": len(text),
                    "label": 0,
                    "estimated_membership": "non_member",
                    "text_hash": h,
                })

                year_count += 1

            print(f"  acumuladas {year}: {year_count}")

        print(f"Total {year}: {year_count}")

    df = pd.DataFrame(metadata)
    df.to_csv(f"{BASE_DIR}/metadata/articulos_historicos_es_metadata.csv", index=False)

    print("\nNoticias históricas exportadas:", len(df))
    if len(df) > 0:
        print(df["year"].value_counts().sort_index())


def main():
    make_dirs()

    wikimia_df = export_wikimia_like("swj0419/WikiMIA", "WikiMIA", "wikimia")
    wikimia24_df = export_wikimia_like("wjfu99/WikiMIA-24", "WikiMIA24", "wikimia24")

    build_wikimia_extended(wikimia_df, wikimia24_df)

    export_wikimia_like("SimMIA/WikiMIA-25", "WikiMIA25", "wikimia25")

    export_booktection()

    export_recent_spanish_news()
    export_historical_spanish_news()

    print("\nTODO LISTO")


if __name__ == "__main__":
    main()