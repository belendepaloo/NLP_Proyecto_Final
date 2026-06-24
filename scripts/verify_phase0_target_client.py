#!/usr/bin/env python3
"""
verify_phase0_target_client.py — milestone demostrable de la Fase 0 del plan de
ingenieria: probar que DE-COP, SiMIA y DUALTEST corren los tres contra el MISMO
cliente target (por defecto, Llama via Groq), fuera de cualquier agente/web.

No corre el pipeline completo (scraping/curacion/chunking/SAGE -- eso es Fase 1).
Solo valida la unificacion del cliente target descripta en mia_common/target_client.py.

Uso:
    export GROQ_API_KEY=...
    python scripts/verify_phase0_target_client.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mia_common.settings import settings  # noqa: E402
from mia_common.target_client import as_dualtest_target, make_target_client  # noqa: E402

SAMPLE_VERBATIM = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, "
    "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
    "incredulity."
)
# Candidatos de paraphrase de muestra -- en la Fase 1 estos van a venir de
# SAGE.paraphrase(text)["segments"][i]["all_candidates"], no hardcodeados.
SAMPLE_PARAPHRASES = [
    "Those years held both the greatest joys and the deepest sorrows; they were marked "
    "by profound understanding as much as by utter folly, by faith as much as by doubt.",
    "It stood as a time of extremes, brilliant and bleak together, a season of insight "
    "shadowed by foolishness, of conviction tangled with disbelief.",
    "That period contained every contradiction: triumph and despair, wisdom and "
    "madness, deep belief and equally deep skepticism, all at once.",
]
BOOK_TITLE = "A Tale of Two Cities"
AUTHOR = "Charles Dickens"
NON_MEMBER_PREFIX = "The quarterly earnings report exceeded analyst expectations this fiscal year."


def main() -> int:
    if not settings.groq_api_key:
        print(
            "GROQ_API_KEY no esta configurada (.env o variable de entorno).\n"
            "Conseguila en https://console.groq.com y exportala como GROQ_API_KEY "
            "antes de correr este script (ver .env.example).",
            file=sys.stderr,
        )
        return 1

    client = make_target_client(
        provider=settings.target_provider,
        model_name=settings.target_model_name,
        api_key=settings.groq_api_key,
        min_seconds_between_calls=settings.target_min_seconds_between_calls,
        max_retries=settings.target_max_retries,
    )
    print(f"[1/4] Cliente target listo: {settings.target_provider}/{settings.target_model_name}")

    from DE_COP.decop import decop_score

    decop_result = decop_score(
        verbatim_passage=SAMPLE_VERBATIM,
        paraphrase_candidates=SAMPLE_PARAPHRASES,
        book_title=BOOK_TITLE,
        author=AUTHOR,
        client=client,
        n_permutations=3,
    )
    print(
        f"[2/4] DE-COP OK -- accuracy={decop_result['accuracy']:.2f} "
        f"({decop_result['n_queries']} queries)"
    )

    from SiMIA.simia import simmia_score

    simia_result = simmia_score(
        text=SAMPLE_VERBATIM,
        client=client,
        non_member_prefix=NON_MEMBER_PREFIX,
        n_samples=1,
        max_words=10,
    )
    print(f"[3/4] SiMIA OK -- score={simia_result}")

    from DUALTEST.reference_model import ReferenceModel
    from DUALTEST.scoring import score_texts

    print(f"      Cargando modelo de referencia local ({settings.reference_model_name})...")
    reference = ReferenceModel(settings.reference_model_name, device=None)
    dualtest_target = as_dualtest_target(client, max_new_tokens=32)
    df = score_texts(
        texts=[SAMPLE_VERBATIM],
        target=dualtest_target,
        reference=reference,
        prefix_len=50,
        continuation_len=32,
        max_new_tokens=32,
        label=1,
        dataset_name="phase0_smoke_test",
    )
    print("[4/4] DUALTEST OK (mismo cliente target, via as_dualtest_target):")
    print(df[["run_length", "p_rlb", "edit_similarity", "p_esb"]].to_string(index=False))

    print(
        "\nFase 0 verificada: DE-COP, SiMIA y DUALTEST corrieron contra el mismo "
        "cliente target unificado."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
