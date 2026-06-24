"""
decop.py — version importable de DE-COP (Duarte et al., 2024), portada desde el
notebook de Colab DE-COP/DE_COP_BookTection.ipynb (branch feature/decop).

Diferencia clave con el notebook: el notebook arma las 4 opciones desde columnas fijas
de BookTection (Example_A..D). Aca, `decop_score` recibe el pasaje verbatim y los
candidatos de paraphrase generados por SAGE para ese mismo chunk -- asi DE-COP corre
sobre texto scrapeado en vivo, no solo sobre el benchmark.

Nota: el directorio se llama DE_COP (guion bajo) en vez de DE-COP (guion medio, como en
la branch feature/decop) porque un guion medio no es un identificador de Python valido
para `import`. El notebook original, una vez mergeado, queda en DE-COP/ solo como
referencia de evaluacion contra BookTection; este modulo es el que usa el pipeline.
"""

from __future__ import annotations

import itertools
import random
import re

from mia_common.target_client import DailyCapError, TargetClient  # re-exportado para callers

LETTERS = ["A", "B", "C", "D"]

QA_SYS = (
    "You are entering a multiple choice questions exam. You should directly answer "
    "each question by choosing the correct option. Be concise and straight to the "
    "point in your answer. Output only the letter corresponding to the correct answer."
)


def build_prompt(options: list[str], book: str, author: str) -> str:
    q = (
        f'Question: Which of the following passages is verbatim from the "{book}" '
        f"book by {author}?\nOptions:\n"
    )
    body = "".join(f"{letter}. {opt}\n" for letter, opt in zip(LETTERS, options))
    return q + body + "Answer:"


def parse_letter(text: str | None) -> str:
    m = re.search(r"[ABCD]", (text or "").upper())
    return m.group(0) if m else "A"  # fallback, igual que el notebook original


def ask_model(options: list[str], book: str, author: str, client: TargetClient, seed: int = 0) -> str:
    prompt = build_prompt(options, book, author)
    out = client.chat(
        [{"role": "system", "content": QA_SYS}, {"role": "user", "content": prompt}],
        max_new_tokens=2,
        temperature=0,
        seed=seed,
    )
    return parse_letter(out)


def make_permutations(
    options: list[str], n_perms: int | None, rng: random.Random
) -> list[tuple[list[str], str]]:
    """options[0] = pasaje verbatim original; [1..3] = paraphrase candidates de SAGE.
    Devuelve [(opciones_barajadas, letra_correcta), ...]."""
    all_perms = list(itertools.permutations(range(4)))  # 24
    rng.shuffle(all_perms)
    chosen = all_perms if (n_perms is None or n_perms >= 24) else all_perms[:n_perms]
    return [([options[i] for i in p], LETTERS[p.index(0)]) for p in chosen]


def decop_score(
    verbatim_passage: str,
    paraphrase_candidates: list[str],
    book_title: str,
    author: str,
    client: TargetClient,
    n_permutations: int = 6,
    seed: int = 0,
) -> dict:
    """Multiple-choice MIA test: el modelo tiene que identificar cual de las 4 opciones
    es el pasaje VERBATIM (las otras 3 son paraphrases de SAGE). Se promedia sobre
    `n_permutations` ordenes de las opciones para no sesgar por posicion.

    Requiere >=3 paraphrase candidates (de SAGE.paraphrase(...)["segments"][i]
    ["all_candidates"]) -- si no hay suficientes, levanta ValueError; el caller
    (agents/tools/mia_tools.py) debe atajarlo y skippear DE-COP para ese chunk en vez
    de romper el run completo (y loggearlo en la skill de aprendizajes).

    Devuelve {"accuracy": float en [0,1], "n_queries": int, "per_permutation": [...]}.
    accuracy=0.25 es el nivel de azar (4 opciones), no 0 -- el ensemble debe tenerlo en
    cuenta al ponderar, no asumir que 0.25 significa "sin evidencia de membership".
    """
    if len(paraphrase_candidates) < 3:
        raise ValueError(
            "decop_score necesita >=3 paraphrase candidates "
            f"(recibio {len(paraphrase_candidates)}); pedile mas candidatos a SAGE "
            "para este chunk o skippea DE-COP."
        )

    options = [verbatim_passage] + paraphrase_candidates[:3]
    rng = random.Random(seed)

    correct = 0
    total = 0
    per_permutation = []
    for shuffled_options, gold_letter in make_permutations(options, n_permutations, rng):
        answer = ask_model(shuffled_options, book_title, author, client, seed=seed)
        is_correct = answer == gold_letter
        correct += int(is_correct)
        total += 1
        per_permutation.append({"answer": answer, "gold": gold_letter, "correct": is_correct})

    return {
        "accuracy": correct / total,
        "n_queries": total,
        "per_permutation": per_permutation,
    }
