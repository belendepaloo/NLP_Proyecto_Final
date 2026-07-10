import pandas as pd
from pathlib import Path

try:
    # import como paquete (uso normal desde la raiz del repo, ej. agents/tools/)
    from DUALTEST.prefixing import split_by_tokens, split_by_words
    from DUALTEST.metrics import rlb_score, esb_score, esb_score_log
except ImportError:
    # fallback para notebooks existentes que hacen sys.path.append(".../DUALTEST")
    # y despues `from scoring import score_texts` (ver notebooks/dualtest_experiments/)
    from prefixing import split_by_tokens, split_by_words
    from metrics import rlb_score, esb_score, esb_score_log


def _split_for_target(text, target, reference, prefix_len=64, continuation_len=64):
    """
    Si el target tiene tokenizer local, usamos tokens.
    Si el target es API black-box, usamos palabras para construir el prefix
    y después tokenizamos con el reference para calcular métricas.
    """

    if getattr(target, "has_tokenizer", False):
        return split_by_tokens(
            text,
            tokenizer=target.tokenizer,
            prefix_len=prefix_len,
            continuation_len=continuation_len,
        )

    split_words = split_by_words(
        text,
        prefix_len=50,
        continuation_len=continuation_len,
    )

    prefix_ids = reference.tokenizer(
        split_words.prefix_text,
        add_special_tokens=False,
    ).input_ids

    continuation_ids = reference.tokenizer(
        split_words.continuation_text,
        add_special_tokens=False,
    ).input_ids[:continuation_len]

    split_words.prefix_token_ids = prefix_ids
    split_words.continuation_token_ids = continuation_ids

    return split_words


def score_texts(
    texts,
    target,
    reference,
    prefix_len=64,
    continuation_len=64,
    max_new_tokens=64,
    label=0,
    dataset_name="dataset",
):
    """
    target con tokenizer (HFLocalTarget, has_tokenizer=True): prefijo/continuacion se
    cortan por TOKENS del propio target, fiel al paper.

    target sin tokenizer (APITarget, has_tokenizer=False -- API cerrada tipo
    Groq/OpenAI/Anthropic): se cae al fallback de 50 palabras (split_by_words, el mismo
    que usa el Apendice B del paper para GPT-4) y se re-tokeniza prefijo, continuacion
    fuente y la completion del target con el tokenizer del modelo de REFERENCIA -- es
    el unico tokenizer white-box disponible en ese caso, y es justo lo que pide el
    docstring de prefixing.py ("re-tokenizar el mismo string con cada modelo por
    separado").
    """
    rows = []

    for i, text in enumerate(texts):

        split = _split_for_target(
            text=text,
            target=target,
            reference=reference,
            prefix_len=prefix_len,
            continuation_len=continuation_len,
        )

        completion = target.complete(
            split.prefix_text,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        if completion.token_ids is None:
            target_tokens = reference.tokenizer(
                completion.text,
                add_special_tokens=False,
            ).input_ids[:max_new_tokens]
        else:
            target_tokens = completion.token_ids[:max_new_tokens]

        r, p_rlb = rlb_score(
            target_tokens=target_tokens,
            source_tokens=split.continuation_token_ids,
            reference_model=reference,
            prefix_token_ids=split.prefix_token_ids,
        )

        s, p_esb = esb_score(
            target_text=completion.text,
            target_tokens=target_tokens,
            source_text=split.continuation_text,
            reference_model=reference,
            prefix_token_ids=split.prefix_token_ids,
        )

        _, log_p_esb = esb_score_log(
            target_text=completion.text,
            target_tokens=target_tokens,
            source_text=split.continuation_text,
            reference_model=reference,
            prefix_token_ids=split.prefix_token_ids,
        )

        rows.append({
            "id": f"{dataset_name}_{i}",
            "label": label,
            "run_length": r,
            "p_rlb": p_rlb,
            "edit_similarity": s,
            "p_esb": p_esb,
            "log_p_esb": log_p_esb,
            "prefix": split.prefix_text,
            "ground_truth": split.continuation_text,
            "target_completion": completion.text,
        })

    return pd.DataFrame(rows)


def score_records(
    records,
    target,
    reference,
    prefix_len=64,
    continuation_len=64,
    max_new_tokens=64,
):
    texts = [r["text"] for r in records]

    df = score_texts(
        texts=texts,
        target=target,
        reference=reference,
        prefix_len=prefix_len,
        continuation_len=continuation_len,
        max_new_tokens=max_new_tokens,
        label=None,
        dataset_name=records[0]["dataset"] if records else "dataset",
    )

    df["id"] = [r["id"] for r in records]
    df["dataset"] = [r["dataset"] for r in records]
    df["label"] = [r["label"] for r in records]
    df["membership"] = [r["estimated_membership"] for r in records]

    return df

def score_records_with_checkpoint(
    records,
    reference,
    completion_fn,
    output_path,
    target_model_name=None,
    reference_model_name="EleutherAI/pythia-410m",
    prefix_words=50,
    continuation_len=64,
    max_new_tokens=64,
):
    output_path = Path(output_path)

    if output_path.exists():
        df_done = pd.read_csv(output_path)
        done_ids = set(df_done["id"].astype(str))
        rows = df_done.to_dict("records")
        print(f"Retomando checkpoint: {len(done_ids)} ya procesados")
    else:
        done_ids = set()
        rows = []

    for idx, record in enumerate(records):
        record_id = str(record["id"])

        if record_id in done_ids:
            continue

        try:
            split = split_by_words(
                record["text"],
                prefix_len=prefix_words,
                continuation_len=continuation_len,
            )

            prefix_ids = reference.tokenizer(
                split.prefix_text,
                add_special_tokens=False,
            ).input_ids

            source_tokens = reference.tokenizer(
                split.continuation_text,
                add_special_tokens=False,
            ).input_ids[:continuation_len]

            completion_text = completion_fn(
                split.prefix_text,
                max_new_tokens=max_new_tokens,
            )

            target_tokens = reference.tokenizer(
                completion_text,
                add_special_tokens=False,
            ).input_ids[:max_new_tokens]

            r, p_rlb = rlb_score(
                target_tokens=target_tokens,
                source_tokens=source_tokens,
                reference_model=reference,
                prefix_token_ids=prefix_ids,
            )

            s, p_esb = esb_score(
                target_text=completion_text,
                target_tokens=target_tokens,
                source_text=split.continuation_text,
                reference_model=reference,
                prefix_token_ids=prefix_ids,
            )

            log_p_esb = reference.sequence_log_probability(
                prefix_ids,
                target_tokens,
            )

            rows.append({
                "id": record["id"],
                "dataset": record["dataset"],
                "label": record["label"],
                "membership": record["estimated_membership"],
                "prefix": split.prefix_text,
                "ground_truth": split.continuation_text,
                "target_completion": completion_text,
                "run_length": r,
                "p_rlb": p_rlb,
                "edit_similarity": s,
                "p_esb": p_esb,
                "log_p_esb": log_p_esb,
                "target_model": target_model_name,
                "reference_model": reference_model_name,
            })

            done_ids.add(record_id)

            # checkpoint por cada fila
            pd.DataFrame(rows).to_csv(output_path, index=False)

            if len(rows) % 10 == 0:
                print(f"Guardados {len(rows)} / {len(records)}")

        except Exception as e:
            print(f"Error en {record_id}: {e}")

            rows.append({
                "id": record["id"],
                "dataset": record.get("dataset"),
                "label": record.get("label"),
                "membership": record.get("estimated_membership"),
                "error": str(e),
                "target_model": target_model_name,
                "reference_model": reference_model_name ,
            })
            done_ids.add(record_id)
            pd.DataFrame(rows).to_csv(output_path, index=False)

    return pd.DataFrame(rows)