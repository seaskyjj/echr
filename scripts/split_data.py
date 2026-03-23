import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_processed(processed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(processed_path)
    required_cols = {"text", "label", "item_id", "respondent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {processed_path}: {sorted(missing)}")
    return df


def make_outer_strata(df: pd.DataFrame, min_count: int = 2) -> pd.Series:
    """
    Use respondent+label when there is enough support for a stable outer split.
    Collapse rare strata into OTHER__label to avoid split failures after country expansion.
    """
    respondent_label = df["respondent"].astype(str) + "__" + df["label"].astype(str)
    counts = respondent_label.value_counts()
    collapsed = respondent_label.where(
        respondent_label.map(counts) >= min_count,
        "OTHER__" + df["label"].astype(str),
    )
    collapsed_counts = collapsed.value_counts()
    if int(collapsed_counts.min()) < 2:
        return df["label"].astype(str)
    return collapsed


def make_inner_strata(df: pd.DataFrame) -> pd.Series:
    """
    Validation/test stratification is kept label-only for robustness.
    After adding more countries, respondent+label strata become too sparse in the temp split.
    """
    return df["label"].astype(str)


def stratified_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
):
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    strata = make_outer_strata(df)
    outer_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=seed,
    )
    train_idx, temp_idx = next(outer_split.split(df, strata))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    temp_val_share = val_size / (val_size + test_size)
    temp_strata = make_inner_strata(temp_df)
    inner_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=temp_val_share,
        random_state=seed,
    )
    val_idx, test_idx = next(inner_split.split(temp_df, temp_strata))

    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def join_test_metadata(test_df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_path)
    if "itemid" not in metadata_df.columns:
        raise ValueError(f"Missing itemid column in {metadata_path}")

    test_df = test_df.copy()
    metadata_df = metadata_df.copy()
    test_df["item_id"] = test_df["item_id"].astype(str)
    metadata_df["itemid"] = metadata_df["itemid"].astype(str)

    joined_df = test_df.merge(
        metadata_df,
        left_on="item_id",
        right_on="itemid",
        how="left",
        sort=False,
        suffixes=("", "_meta"),
    )
    return joined_df


def summarize_split(name: str, df: pd.DataFrame) -> dict:
    label_counts = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
    respondent_counts = {str(k): int(v) for k, v in df["respondent"].value_counts().sort_index().items()}
    strata_counts = {
        str(k): int(v) for k, v in make_outer_strata(df).value_counts().sort_index().items()
    }
    multi_count = int(df["is_multi_respondent"].fillna(False).astype(bool).sum()) if "is_multi_respondent" in df.columns else 0
    return {
        "split": name,
        "rows": int(len(df)),
        "label_counts": label_counts,
        "respondent_counts": respondent_counts,
        "strata_counts": strata_counts,
        "multi_respondent_rows": multi_count,
    }


def resolve_processed_path(args) -> Path:
    if args.processed_path:
        return Path(args.processed_path)
    return Path(args.dataset_dir) / "data" / "processed" / "processed.csv"


def resolve_metadata_path(args) -> Path:
    if args.metadata_path:
        return Path(args.metadata_path)
    return Path(args.dataset_dir) / "data" / "raw" / "metadata.csv"


def resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.working_dir) / "data" / "processed"


def main(args):
    processed_path = resolve_processed_path(args)
    metadata_path = resolve_metadata_path(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed(processed_path)
    train_df, val_df, test_df = stratified_split(
        df=df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    metadata_joined_test_path = output_dir / "metadata_joined_test.csv"
    summary_path = output_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata_joined_test = join_test_metadata(test_df, metadata_path)
    metadata_joined_test.to_csv(metadata_joined_test_path, index=False)

    summary = {
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "source_rows": int(len(df)),
        "splits": [
            summarize_split("train", train_df),
            summarize_split("val", val_df),
            summarize_split("test", test_df),
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved train split to {train_path}")
    print(f"Saved val split to {val_path}")
    print(f"Saved test split to {test_path}")
    print(f"Saved joined test metadata to {metadata_joined_test_path}")
    print(f"Saved split summary to {summary_path}")
    for split_summary in summary["splits"]:
        print(
            f"{split_summary['split']}: "
            f"rows={split_summary['rows']}, "
            f"labels={split_summary['label_counts']}, "
            f"respondents={split_summary['respondent_counts']}"
        )


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
        "--processed_path",
        type=str,
        default=None,
        help="Override path to processed.csv.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Override path to the raw metadata CSV for joining the test split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override directory where train/val/test files will be written.",
    )
    parser.add_argument("--train_size", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    main(parser.parse_args())
