# Valida el pipeline de Vertex AI con 5-10 muestras de wikimia

# Uso:
#     Desde la raíz del proyecto (donde está la carpeta dataset/)
#     python SAGE/test_vertex_pipeline.py \
#         --model gemini \
#         --project NLP-SAGE-gcp \
#         --n_samples 5

#     Con DeepSeek o Grok:
#     python SAGE/test_vertex_pipeline.py --model deepseek --project NLP-SAGE-gcp
#     python SAGE/test_vertex_pipeline.py --model grok --project NLP-SAGE-gcp


import argparse
import os
import sys
import time
import pandas as pd

# Asegurar que los módulos de SAGE sean importables
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from SAGE.paraphraser import Paraphraser, VERTEX_MODEL_IDS, ALL_VERTEX_ALIASES


# CLI                                                                     

def parse_args():
    parser = argparse.ArgumentParser(description="Test Vertex AI paraphrase pipeline")
    parser.add_argument(
        "--model",
        choices=list(ALL_VERTEX_ALIASES.keys()),
        default="gemini",
        help="Modelo Vertex AI a probar (gemini | deepseek | grok)",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP Project ID",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Región de Vertex AI (default: us-central1)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Cantidad de muestras a procesar (default: 5)",
    )
    parser.add_argument(
        "--metadata",
        default="dataset/metadata/wikimia_metadata.csv",
        help="Ruta al CSV de metadata",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=3,
        help="Candidatos por segmento (default: 3, igual que T5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta del CSV de salida. Por defecto: dataset/sage_outputs/<model>/wikimia_test_<model>.csv",
    )
    return parser.parse_args()


# Pipeline de prueba (sin SPS/WordSim para mantenerlo liviano)          
def run_test(args):
    print("=" * 60)
    print(f"SAGE - Vertex AI pipeline test")
    print(f"Modelo  : {args.model}  ({ALL_VERTEX_ALIASES[args.model]})")
    print(f"Proyecto: {args.project}")
    print(f"Región  : {args.location}")
    print(f"Muestras: {args.n_samples}")
    print("=" * 60)

    # Cargar metadata                                                      
    if not os.path.exists(args.metadata):
        print(f"[ERROR] No se encontró el CSV de metadata: {args.metadata}")
        print("Asegurate de correr el script desde la raíz del proyecto.")
        sys.exit(1)

    df = pd.read_csv(args.metadata)
    df_sample = df.sample(n=min(args.n_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"\nMuestras seleccionadas: {len(df_sample)}")


    # Inicializar paraphraser                                              
    try:
        paraphraser = Paraphraser(
            provider="vertex_ai",
            model_name=args.model,
            gcp_project=args.project,
            gcp_location=args.location,
        )
    except Exception as e:
        print(f"\n[ERROR] No se pudo inicializar Vertex AI: {e}")
        print("\nChecklist:")
        print("1. Habilitaste la API de Vertex AI en GCP.")
        print("2. Seteaste GOOGLE_APPLICATION_CREDENTIALS con la ruta a tu key JSON.")
        print("Ej: export GOOGLE_APPLICATION_CREDENTIALS=/ruta/a/key.json")
        print("3. El Service Account tiene rol 'Vertex AI User'.")
        print("4. Habilitaste el modelo en Model Garden (deepseek / grok).")
        sys.exit(1)


    # Procesar muestras                                                   
    results = []
    errors = []

    for i, row in df_sample.iterrows():
        label = f"[{i+1}/{len(df_sample)}]"
        file_path = row.get("file_path", "")
        file_name = row.get("file_name", os.path.basename(file_path))
        membership = row.get("estimated_membership", "?")

        print(f"\n{label} {membership} - {file_name}")

        if not os.path.exists(file_path):
            print(f"[SKIP] Archivo no encontrado: {file_path}")
            errors.append({"file_name": file_name, "error": "file not found"})
            continue

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        print(f"  Original (200c): {text[:200].replace(chr(10), ' ')}")

        t0 = time.time()
        try:
            candidates = paraphraser.generate_candidates(text[:1000], n=args.n_candidates)
            elapsed = time.time() - t0

            if candidates:
                best = candidates[0]
                print(f"  Paráfrasis (200c): {best[:200].replace(chr(10), ' ')}")
                print(f"  Candidatos: {len(candidates)}  |  Tiempo: {elapsed:.1f}s")
                results.append({
                    "file_name": file_name,
                    "membership": membership,
                    "original_preview": text[:300],
                    "paraphrase_1": candidates[0] if len(candidates) > 0 else "",
                    "paraphrase_2": candidates[1] if len(candidates) > 1 else "",
                    "paraphrase_3": candidates[2] if len(candidates) > 2 else "",
                    "n_candidates": len(candidates),
                    "elapsed_s": round(elapsed, 2),
                    "model": args.model,
                })
            else:
                print("  [WARN] Sin candidatos devueltos.")
                errors.append({"file_name": file_name, "error": "no candidates"})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"[ERROR] {e} ({elapsed:.1f}s)")
            errors.append({"file_name": file_name, "error": str(e)})


    # Guardar resultados                                                   
    output_dir = f"dataset/sage_outputs/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = args.output or f"{output_dir}/wikimia_test_{args.model}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)


    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"OK : {len(results)}")
    print(f"Errores: {len(errors)}")
    if results:
        avg_time = sum(r["elapsed_s"] for r in results) / len(results)
        print(f"Tiempo promedio por muestra: {avg_time:.1f}s")
        print(f"Tiempo estimado para dataset completo (~1000 muestras): "
              f"{avg_time * 1000 / 60:.0f} min")
    print(f"\n Resultados guardados en: {output_path}")

    if errors:
        print(f"\n Errores detallados:")
        for e in errors:
            print(f"{e['file_name']}: {e['error']}")

    print("\n  ✓ Pipeline validado. Listo para escalar." if not errors else
          "\n  ⚠ Hubo errores - revisalos antes de escalar.")

if __name__ == "__main__":
    args = parse_args()
    run_test(args)