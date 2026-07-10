import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss


DEFAULT_FEATURES = [
    "run_length",
    "edit_similarity",
    "esb_evidence",
]


def prepare_probability_features(df, include_semantic=False, require_label=True,):
    df = df.copy()

    if "log_p_esb" not in df.columns:
        raise ValueError("El dataframe necesita columna log_p_esb.")

    df["neg_log_p_esb"] = -df["log_p_esb"]
    df["esb_evidence"] = df["neg_log_p_esb"] * df["edit_similarity"]

    features = DEFAULT_FEATURES.copy()

    if include_semantic:
        if "semantic_similarity" in df.columns:
            features.append("semantic_similarity")
        elif "semantic_similarity_target" in df.columns:
            features.append("semantic_similarity_target")
        else:
            raise ValueError("Pediste semantic pero no hay columna semantic_similarity.")

    if require_label:
        needed = features + ["label"]
    else:
        needed = features
        
    df = df.dropna(subset=needed).copy()

    return df, features


def load_and_merge_csvs(csv_paths):
    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        dfs.append(df)

    if not dfs:
        raise ValueError("csv_paths está vacío.")

    return pd.concat(dfs, ignore_index=True)


def fit_membership_probability_model(
    df,
    include_semantic=False,
    test_size=0.3,
    random_state=7,
    calibration_method="isotonic",
):
    df, features = prepare_probability_features(
        df,
        include_semantic=include_semantic,
    )

    X = df[features]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    base_model = LogisticRegression(max_iter=1000)

    model = CalibratedClassifierCV(
        estimator=base_model,
        method=calibration_method,
        cv=5,
    )

    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "features": features,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc": roc_auc_score(y_test, test_probs),
        "brier": brier_score_loss(y_test, test_probs),
        "mean_probability": float(np.mean(test_probs)),
    }

    df_out = df.copy()
    df_out["membership_probability"] = model.predict_proba(X)[:, 1]

    return model, df_out, metrics


def apply_membership_probability_model(
    df,
    model,
    include_semantic=False,
):
    df, features = prepare_probability_features(
        df,
        include_semantic=include_semantic,
        require_label=False,
    )

    df["membership_probability"] = model.predict_proba(df[features])[:, 1]

    return df


def find_threshold_for_fpr(
    df,
    prob_col="membership_probability",
    label_col="label",
    target_fpr=0.01,
):
    nonmember_probs = df[df[label_col] == 0][prob_col].dropna()

    if len(nonmember_probs) == 0:
        raise ValueError("No hay non-members para calcular FPR.")

    threshold = nonmember_probs.quantile(1 - target_fpr)

    return float(threshold)


def evaluate_threshold(
    df,
    threshold,
    prob_col="membership_probability",
    label_col="label",
):
    d = df.dropna(subset=[prob_col, label_col]).copy()

    y_true = d[label_col].astype(int)
    y_pred = (d[prob_col] >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
    }


def find_and_evaluate_threshold(
    df,
    prob_col="membership_probability",
    label_col="label",
    target_fpr=0.01,
):
    threshold = find_threshold_for_fpr(
        df,
        prob_col=prob_col,
        label_col=label_col,
        target_fpr=target_fpr,
    )

    return evaluate_threshold(
        df,
        threshold,
        prob_col=prob_col,
        label_col=label_col,
    )


def add_suspicious_flag(
    df,
    threshold,
    prob_col="membership_probability",
):
    df = df.copy()
    df["membership_threshold"] = threshold
    df["suspicious"] = df[prob_col] >= threshold
    return df


def calibrate_csv(
    input_csv,
    output_csv=None,
    include_semantic=False,
    test_size=0.3,
    random_state=7,
    target_fpr=0.01,
):
    df = pd.read_csv(input_csv)

    model, df_out, metrics = fit_membership_probability_model(
        df,
        include_semantic=include_semantic,
        test_size=test_size,
        random_state=random_state,
    )

    threshold_report = find_and_evaluate_threshold(
        df_out,
        target_fpr=target_fpr,
    )

    df_out = add_suspicious_flag(
        df_out,
        threshold=threshold_report["threshold"],
    )

    metrics["threshold_report"] = threshold_report

    if output_csv is not None:
        df_out.to_csv(output_csv, index=False)

    return model, df_out, metrics


def calibrate_general_csvs(
    csv_paths,
    output_csv=None,
    include_semantic=False,
    test_size=0.3,
    random_state=7,
    calibration_method="isotonic",
    target_fpr=0.01,
):
    df_all = load_and_merge_csvs(csv_paths)

    model, df_out, metrics = fit_membership_probability_model(
        df_all,
        include_semantic=include_semantic,
        test_size=test_size,
        random_state=random_state,
        calibration_method=calibration_method,
    )

    threshold_report = find_and_evaluate_threshold(
        df_out,
        target_fpr=target_fpr,
    )

    df_out = add_suspicious_flag(
        df_out,
        threshold=threshold_report["threshold"],
    )

    metrics["threshold_report"] = threshold_report
    metrics["csv_paths"] = [str(p) for p in csv_paths]
    metrics["target_fpr"] = target_fpr

    if output_csv is not None:
        df_out.to_csv(output_csv, index=False)

    return model, df_out, metrics