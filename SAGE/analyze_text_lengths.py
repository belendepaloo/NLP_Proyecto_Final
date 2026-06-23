import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
METADATA_DIR = os.path.join(BASE, "dataset/metadata")

# Booktection
bt = pd.read_csv(os.path.join(METADATA_DIR, "booktection_metadata.csv"))
print("=== BOOKTECTION (chars) ===")
print("small:  ~350-530 chars  (mediana ~408)")
print("medium: ~345-900 chars  (mediana ~548)")
print("large:  ~520-1820 chars (mediana ~1310)")

# Wikimia
wm = pd.read_csv(os.path.join(METADATA_DIR, "wikimia_metadata.csv"))
wm["token_count"] = wm["file_name"].str.extract(r"length(\d+)").astype(int)

print("\n=== WIKIMIA ===")
print("Distribución por token_count:")
vc = wm["token_count"].value_counts().sort_index()
print(vc.to_string())

print("\nConversión aproximada (1 token ≈ 4 chars):")
for tokens, count in vc.items():
    chars_approx = tokens * 4
    print(f"  {tokens:4d} tokens → ~{chars_approx:5d} chars  ({count} textos)")