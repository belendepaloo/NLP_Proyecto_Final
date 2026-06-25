#!/usr/bin/env python3
"""
build_simia_calibration_set.py — arma el prefijo non-member FIJO (P) que SimMIA
necesita para la perturbacion (ver paper arXiv:2601.11314, Yi & Li 2026: "fixed
non-member prefixes reduce variance" vs. eleccion aleatoria, sigma~0.86 vs 2.90).

Deliberadamente SEPARADO de processRawText/Datasets/dataset_len128.csv (el dataset de
evaluacion) -- son capitulos de Royal Road distintos a los 5 que se usan como
non-member en la evaluacion, para que el prefijo de perturbacion nunca se solape con
lo que se esta midiendo.

Uso:
    python scripts/build_simia_calibration_set.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processRawText.text_pipeline import clean_text  # noqa: E402

OUT_PATH = ROOT / "SiMIA" / "non_member_calibration.txt"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# 5 capitulos de Royal Road, todos posteriores al cutoff de Llama 3.1 (~dic 2023),
# todos DISTINTOS de los 5 usados como non-member en el dataset de evaluacion.
CALIBRATION_SOURCES = [
    ("https://www.royalroad.com/fiction/21410/super-minion/chapter/2878429/ch53-lore-dumplings", "2025-12-25"),
    (
        "https://www.royalroad.com/fiction/107917/sky-pride/chapter/3589044/chapter-19--a-quiet-word-with-a-heretic",
        "2026-06-25",
    ),
    (
        "https://www.royalroad.com/fiction/62125/ghost-in-the-city-cyberpunk-gamer-si/chapter/3431124/chapter-245",
        "2026-05-22",
    ),
    (
        "https://www.royalroad.com/fiction/48402/magical-girl-gunslinger/chapter/2215348/chapter-37-reflection",
        "2025-04-19",
    ),
    ("https://www.royalroad.com/fiction/39408/beware-of-chicken/chapter/3582621/v7c70-tribulation", "2026-06-24"),
]

MAX_CHARS_PER_SOURCE = 2000  # un fragmento por fuente alcanza; no hace falta el capitulo entero


def main() -> None:
    excerpts = []
    for url, date in CALIBRATION_SOURCES:
        resp = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=30)
        resp.raise_for_status()
        cleaned = clean_text(resp.text, is_html=True)
        excerpts.append(cleaned[:MAX_CHARS_PER_SOURCE])
        print(f"[{date}] {url} -> {len(cleaned)} chars limpios (usando {MAX_CHARS_PER_SOURCE})")

    calibration_text = "\n\n".join(excerpts)
    OUT_PATH.write_text(calibration_text, encoding="utf-8")
    print(f"\n{len(excerpts)} fuentes, {len(calibration_text)} chars totales -> {OUT_PATH}")


if __name__ == "__main__":
    main()
