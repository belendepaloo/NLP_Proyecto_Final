import sys
sys.path.insert(0, "SAGE")

import os
import pandas as pd

from SAGE.sage import SAGE
from SAGE.paraphraser import Paraphraser


N_SAMPLES = 3
METADATA = "dataset/metadata/wikimia_metadata.csv"

PROJECT = "nlp-sage"
LOCATION = "us-central1"

df = pd.read_csv(METADATA)

# print(df.iloc[0]["file_path"])

PRICE_PER_M_INPUT = 0.30
PRICE_PER_M_OUTPUT = 2.50

def estimated_cost(usage_stats):
    return (
        usage_stats["prompt_tokens"] / 1_000_000 * PRICE_PER_M_INPUT
        + usage_stats["candidates_tokens"] / 1_000_000 * PRICE_PER_M_OUTPUT
    )

def main():

    df = pd.read_csv(METADATA)
    df = df.sample(n=min(N_SAMPLES, len(df)), random_state=42)

    sage = SAGE()

    sage.paraphraser = Paraphraser(
        provider="vertex_ai",
        model_name="gemini",
        gcp_project=PROJECT,
        gcp_location=LOCATION,
    )

    for i, row in enumerate(df.itertuples(), start=1):

        print("\n" + "=" * 80)
        print(f"[{i}/{len(df)}]")

        file_path = row.file_path

        if not os.path.exists(file_path):
            print("Archivo no encontrado:", file_path)
            continue

        # print("PATH:", file_path)

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        print("FILE LEN:", len(text))
        # print("CONTENT:")
        # print(text)

        # text = text[:1000]
        print(len(text))
        result = sage.paraphrase(text)

        print("\nORIGINAL:")
        print(result["original"][:300])

        print("\nPARAFRASIS FINAL:")
        print(result["paraphrase"][:300])

        print("\nDETALLE DE CANDIDATOS:")

        for seg in result["segments"]:

            if seg["type"] != "narrative":
                continue

            print("\n--- Segmento ---")
            print("Original:")
            print(seg["original"][:200])

            print("\nCANDIDATOS GENERADOS:")
            best_score = seg["final_score"]
            for j, cand in enumerate(seg["all_candidates"], start=1):
                winner = ""
                if abs(cand["final_score"] - best_score) < 1e-8:
                    winner = " <-- GANADOR"
                print(f"\nCandidate {j}{winner}")
                print(cand["text"][:200])
                print("SPS      =", round(cand["sps"], 4))
                print("WordSim  =", round(cand["wordsim"], 4))
                print("Final    =", round(cand["final_score"], 4))

            print("\nGANADOR:")
            print(seg["selected"][:200])

            print("\nSPS:", round(seg["sps"], 4))
            print("WordSim:", round(seg["wordsim"], 4))
            print("Final:", round(seg["final_score"], 4))

        cost_so_far = estimated_cost(sage.paraphraser.usage_stats)
        print(f"\n[Costo acumulado estimado: ${cost_so_far:.4f} USD | "
              f"llamadas={sage.paraphraser.usage_stats['calls']} | "
              f"reintentos={sage.paraphraser.usage_stats['retries']}]")

    print("\n✓ Test finalizado")
    print(f"Resumen de uso: {sage.paraphraser.usage_stats}")
    print(f"Costo total estimado: ${estimated_cost(sage.paraphraser.usage_stats):.4f} USD")


if __name__ == "__main__":
    main()