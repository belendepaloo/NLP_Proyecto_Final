#!/usr/bin/env python3
"""
expand_dataset.py — amplia processRawText/Datasets/dataset_len128.csv a 5 textos
member + 5 non-member, todos NARRATIVOS (no listas de citas/resumenes), para una
comparacion member-vs-non-member de manzanas con manzanas.

Member (label=1): 5 novelas de Project Gutenberg, dominio publico, presentes en el
training set de practicamente cualquier LLM (3 ya estaban: A Tale of Two Cities,
Great Expectations, Oliver Twist; se agregan Pride and Prejudice y Frankenstein).

Non-member (label=0): 5 capitulos de novelas serializadas de Royal Road, publicados
en junio 2026 -- muy posteriores al cutoff de entrenamiento de Llama 3.1 (~dic 2023),
asi que no pueden estar en el training set del target. Reemplazan al articulo de
Wikipedia "2025 in video games" (que era listas de citas/obituarios, no narrativa) que
estaba antes como unico non-member.

Uso:
    python scripts/expand_dataset.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processRawText.text_pipeline import build_chunk_dataset, make_tiktoken_counter  # noqa: E402
RAW_DIR = ROOT / "processRawText" / "Raw Texts"
OUT_DIR = ROOT / "processRawText" / "Datasets"
MANIFEST = RAW_DIR / "_manifest.csv"

GUTENBERG_UA = "Mozilla/5.0 (research project)"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Los 3 members existentes se mantienen (ya estan en Raw Texts/ con su entrada en el
# manifest); aca solo se agregan los 2 nuevos members y los 5 non-members narrativos.
NEW_SOURCES = [
    # --- members nuevos (Gutenberg, dominio publico) ---
    {
        "file": "Pride_and_Prejudice.html",
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342-images.html",
        "title": "Pride and Prejudice",
        "date": "1813",
        "label": 1,
        "ua": GUTENBERG_UA,
    },
    {
        "file": "Frankenstein.html",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84-images.html",
        "title": "Frankenstein",
        "date": "1818",
        "label": 1,
        "ua": GUTENBERG_UA,
    },
    # --- non-members narrativos (Royal Road, capitulos de junio 2026) ---
    {
        "file": "RR_GameAtCarousel.html",
        "url": (
            "https://www.royalroad.com/fiction/65629/the-game-at-carousel-book-four-stubs-july-17th"
            "/chapter/3586429/book-nine-chapter-5-dead-pursuit"
        ),
        "title": "The Game at Carousel - Book Nine Chapter 5",
        "date": "2026-06-24",
        "label": 0,
        "ua": BROWSER_UA,
    },
    {
        "file": "RR_ThroneOfTime.html",
        "url": (
            "https://www.royalroad.com/fiction/172653/throne-of-time-magical-academy-time-loop-mystery"
            "/chapter/3585367/chapter-14-first-term-feast"
        ),
        "title": "Throne of Time - Chapter 14",
        "date": "2026-06-24",
        "label": 0,
        "ua": BROWSER_UA,
    },
    {
        "file": "RR_HeadlessOverHeels.html",
        "url": (
            "https://www.royalroad.com/fiction/173313/headless-over-heels-a-dark-fantasy-romancebook"
            "/chapter/3588765/127-blackened-heart"
        ),
        "title": "Headless Over Heels - Chapter 127",
        "date": "2026-06-25",
        "label": 0,
        "ua": BROWSER_UA,
    },
    {
        "file": "RR_ClaraCasewell.html",
        "url": (
            "https://www.royalroad.com/fiction/151748/clara-casewell-attorney-to-the-villainess-vol"
            "/chapter/3582863/book-2-chapter-5-esperanca"
        ),
        "title": "Clara Casewell - Book 2 Chapter 5",
        "date": "2026-06-24",
        "label": 0,
        "ua": BROWSER_UA,
    },
    {
        "file": "RR_LegendOfWilliamOh.html",
        "url": (
            "https://www.royalroad.com/fiction/92144/the-legend-of-william-oh"
            "/chapter/3586525/chapter-290-thats-a-good-spider"
        ),
        "title": "The Legend of William Oh - Chapter 290",
        "date": "2026-06-24",
        "label": 0,
        "ua": BROWSER_UA,
    },
]


def download_new_sources() -> None:
    for src in NEW_SOURCES:
        path = RAW_DIR / src["file"]
        if path.exists():
            print(f"[cache] {src['file']}")
            continue
        print(f"[GET] {src['title']} <- {src['url']}")
        resp = requests.get(src["url"], headers={"User-Agent": src["ua"]}, timeout=40)
        resp.raise_for_status()
        path.write_text(resp.text, encoding="utf-8")
        time.sleep(1.5)


def rewrite_manifest() -> list[dict]:
    """Mantiene los 3 members existentes (Dickens) y agrega los 7 nuevos. Saca el
    non-member viejo de Wikipedia (no era narrativo)."""
    existing = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "1":  # los 3 Dickens existentes
                existing.append(row)

    new_rows = [
        {"file": f"Raw Texts/{s['file']}", "url": s["url"], "title": s["title"],
         "date": s["date"], "label": s["label"]}
        for s in NEW_SOURCES
    ]
    all_rows = existing + new_rows

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "url", "title", "date", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Manifest: {len(existing)} members existentes + {len(new_rows)} nuevos = {len(all_rows)} fuentes")
    return all_rows


def build_dataset(rows: list[dict], target: int = 128) -> pd.DataFrame:
    counter = make_tiktoken_counter()
    items = []
    for row in rows:
        path = ROOT / "processRawText" / row["file"]
        raw = path.read_text(encoding="utf-8", errors="ignore")
        items.append({"raw": raw, "source": row["title"], "date": row["date"]})
        print(f"  preparado: {row['title']} ({len(raw)} chars crudos)")

    print("Chunkeando todo (puede tardar varios minutos en las novelas completas)...")
    df = build_chunk_dataset(items, target=target, count_fn=counter, is_html=True, dedup=True)

    label_by_source = {row["title"]: int(row["label"]) for row in rows}
    df["label"] = df["source"].map(label_by_source)
    df["url"] = df["source"].map({row["title"]: row["url"] for row in rows})
    df["title"] = df["source"]
    return df[["text", "label", "title", "source", "url", "date", "n_words", "n_tokens", "length_bucket"]]


def main() -> None:
    download_new_sources()
    rows = rewrite_manifest()
    df = build_dataset(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "dataset_len128.csv"
    jsonl_path = OUT_DIR / "dataset_len128.jsonl"
    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"\n{len(df)} chunks totales -> {csv_path}")
    print(df.groupby(["label", "title"]).size())


if __name__ == "__main__":
    main()
