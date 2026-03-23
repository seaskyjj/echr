import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"item_id", "text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def compute_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_candidates(seed: int) -> dict:
    return {
        "naive_bayes_word": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        lowercase=True,
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
        "naive_bayes_char": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        lowercase=True,
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
        "linear_svm_word": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        lowercase=True,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "linear_svm_char": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        lowercase=True,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def score_values(model: Pipeline, texts):
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(classifier, "decision_function"):
        return model.decision_function(texts)
    return None


def save_predictions(model: Pipeline, df: pd.DataFrame, split_name: str, model_name: str, output_dir: Path):
    preds = model.predict(df["text"])
    passthrough_cols = [
        col
        for col in [
            "item_id",
            "label",
            "respondent",
            "year",
            "is_multi_respondent",
            "respondent_list",
            "source_batch",
            "source_type",
        ]
        if col in df.columns
    ]
    pred_df = df[passthrough_cols].copy()
    pred_df["prediction"] = preds
    scores = score_values(model, df["text"])
    if scores is not None:
        pred_df["score"] = scores
    pred_path = output_dir / f"{model_name}_{split_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    return pred_path, compute_metrics(df["label"], preds)


def family_name(model_name: str) -> str:
    if model_name.startswith("naive_bayes"):
        return "naive_bayes"
    if model_name.startswith("linear_svm"):
        return "linear_svm"
    raise ValueError(f"Unknown model family for {model_name}")


def resolve_split_dir(args) -> Path:
    if args.split_dir:
        return Path(args.split_dir)
    return Path(args.dataset_dir) / "data" / "processed"


def resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.working_dir) / "results" / "classical"


def main(args):
    data_dir = resolve_split_dir(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_split(data_dir / "train.csv")
    val_df = read_split(data_dir / "val.csv")
    test_df = read_split(data_dir / "test.csv")

    candidates = build_candidates(args.seed)

    selection_rows = []

    for model_name, model in candidates.items():
        model.fit(train_df["text"], train_df["label"])
        _, val_metrics = save_predictions(model, val_df, "val", model_name, output_dir)
        row = {"model_name": model_name, "family": family_name(model_name), **val_metrics}
        selection_rows.append(row)
        print(f"{model_name} val metrics: {val_metrics}")

    selection_df = pd.DataFrame(selection_rows).sort_values(["family", "f1", "accuracy"], ascending=[True, False, False])
    selection_path = output_dir / "model_selection_val.csv"
    selection_df.to_csv(selection_path, index=False)

    best_models = (
        selection_df.groupby("family", sort=True)
        .head(1)
        .reset_index(drop=True)
    )

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    final_rows = []
    chosen = {}

    for _, best_row in best_models.iterrows():
        model_name = best_row["model_name"]
        final_model = build_candidates(args.seed)[model_name]
        final_model.fit(train_val_df["text"], train_val_df["label"])
        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(final_model, model_path)

        pred_path, test_metrics = save_predictions(final_model, test_df, "test", model_name, output_dir)
        final_rows.append(
            {
                "family": family_name(model_name),
                "selected_model": model_name,
                **test_metrics,
                "model_path": str(model_path),
                "prediction_path": str(pred_path),
            }
        )
        chosen[family_name(model_name)] = model_name
        print(f"{model_name} test metrics: {test_metrics}")

    test_metrics_df = pd.DataFrame(final_rows).sort_values("family")
    test_metrics_path = output_dir / "test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False)

    config_path = output_dir / "selected_models.json"
    config_path.write_text(json.dumps({"seed": args.seed, "selected_models": chosen}, indent=2), encoding="utf-8")

    print(f"Saved validation model selection to {selection_path}")
    print(f"Saved final test metrics to {test_metrics_path}")
    print(f"Saved selected model manifest to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=".",
        help="Dataset root containing data/raw and data/processed.",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=".",
        help="Writable root for outputs. Useful on Kaggle, e.g. /kaggle/working.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Override directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override directory where model artifacts and metrics will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    main(parser.parse_args())
