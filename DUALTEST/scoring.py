import pandas as pd

try:
    # import como paquete (uso normal desde la raiz del repo, ej. agents/tools/)
    from DUALTEST.prefixing import split_by_tokens, split_by_words
    from DUALTEST.metrics import rlb_score, esb_score
except ImportError:
    # fallback para notebooks existentes que hacen sys.path.append(".../DUALTEST")
    # y despues `from scoring import score_texts` (ver notebooks/dualtest_experiments/)
    from prefixing import split_by_tokens, split_by_words
    from metrics import rlb_score, esb_score


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

        if target.has_tokenizer:
            split = split_by_tokens(
                text,
                tokenizer=target.tokenizer,
                prefix_len=prefix_len,
                continuation_len=continuation_len,
            )
        else:
            split = split_by_words(text, prefix_len=50, continuation_len=continuation_len)
            split.prefix_token_ids = reference.tokenizer(
                split.prefix_text, add_special_tokens=False
            ).input_ids
            split.continuation_token_ids = reference.tokenizer(
                split.continuation_text, add_special_tokens=False
            ).input_ids

        completion = target.complete(
            split.prefix_text,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        target_token_ids = completion.token_ids
        if target_token_ids is None:
            target_token_ids = reference.tokenizer(
                completion.text, add_special_tokens=False
            ).input_ids

        r, p_rlb = rlb_score(
            target_tokens=target_token_ids,
            source_tokens=split.continuation_token_ids,
            reference_model=reference,
            prefix_token_ids=split.prefix_token_ids,
        )

        s, p_esb = esb_score(
            target_text=completion.text,
            target_tokens=target_token_ids,
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
            "prefix": split.prefix_text,
            "ground_truth": split.continuation_text,
            "target_completion": completion.text,
        })

    return pd.DataFrame(rows)