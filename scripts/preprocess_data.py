import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


TARGET_ARTICLES = ["3", "5", "6", "8"]
TARGET_COUNTRIES = ["RUS", "TUR", "GBR"]


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_facts(text: str) -> str:
    """Extract FACTS section from an ECHR judgment."""
    text_upper = str(text).upper()

    start_markers = [
        "AS TO THE FACTS",
        "THE FACTS",
        "I.  THE CIRCUMSTANCES OF THE CASE",
        "THE CIRCUMSTANCES OF THE CASE",
    ]

    start_idx = -1
    for marker in start_markers:
        idx = text_upper.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break
    if start_idx == -1:
        return ""

    end_markers = [
        "THE LAW",
        "RELEVANT DOMESTIC LAW",
        "RELEVANT LEGAL FRAMEWORK",
        "PROCEEDINGS BEFORE THE COMMISSION",
    ]

    best_end_idx = len(text_upper)
    for marker in end_markers:
        idx = text_upper.find(marker, start_idx)
        if idx != -1 and idx < best_end_idx:
            best_end_idx = idx

    return str(text)[start_idx:best_end_idx].strip()


def normalize_respondent(respondent: str, target_countries: List[str] = TARGET_COUNTRIES) -> str:
    """
    Map multi-respondent strings like 'MDA;RUS' to one target country in priority order.
    """
    s = str(respondent).strip()
    if ";" in s:
        for country in target_countries:
            if country in s:
                return country
    return s


def extract_target_articles(field_val: str, target_articles: List[str] = TARGET_ARTICLES) -> List[str]:
    """
    Exact match:
    - match `3` but not `13`
    - allow article suffixes like `3-1`
    """
    s = str(field_val).strip().lower()
    if not s or s in {"nan", "false", "0", ""}:
        return []

    found = []
    for art in target_articles:
        if re.search(rf"(?:^|;){re.escape(art)}(?:$|;|-)", s):
            found.append(art)
    return found


def classify_case(row: Dict, target_articles: List[str] = TARGET_ARTICLES) -> Tuple[str, List[str], List[str]]:
    """
    Return:
    - violation / non-violation / mixed / out_of_scope
    - matched violation articles
    - matched non-violation articles
    """
    v_arts = extract_target_articles(row.get("violation", ""), target_articles)
    nv_arts = extract_target_articles(row.get("nonviolation", ""), target_articles)

    if v_arts and not nv_arts:
        return "violation", v_arts, nv_arts
    if nv_arts and not v_arts:
        return "non-violation", v_arts, nv_arts
    if v_arts and nv_arts:
        return "mixed", v_arts, nv_arts
    return "out_of_scope", v_arts, nv_arts


def load_full_text_map(full_text_path: str) -> Dict[str, str]:
    with open(full_text_path, "r", encoding="utf-8") as f:
        full_texts = json.load(f)

    text_map = {}
    for item in full_texts:
        iid = str(item.get("item_id", "")).strip()
        if not iid:
            continue
        text_map[iid] = item.get("full_text", "")
    return text_map


def preprocess(data_dir: str, min_facts_chars: int = 100, keep_mixed: bool = False):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    full_text_path = os.path.join(data_dir, "full_text.json")
    if not os.path.exists(metadata_path) or not os.path.exists(full_text_path):
        raise FileNotFoundError(f"Missing metadata/full_text in {data_dir}")

    df_meta = pd.read_csv(metadata_path)
    text_map = load_full_text_map(full_text_path)

    stats = {
        "total_meta_rows": len(df_meta),
        "missing_text_entry": 0,
        "empty_full_text": 0,
        "facts_not_found_or_short": 0,
        "skipped_mixed": 0,
        "out_of_scope_or_unlabeled": 0,
        "kept": 0,
    }

    processed_data = []
    for _, row in df_meta.iterrows():
        item_id = str(row.get("itemid", "")).strip()
        if item_id not in text_map:
            stats["missing_text_entry"] += 1
            continue

        full_text = text_map[item_id]
        if not full_text:
            stats["empty_full_text"] += 1
            continue

        facts = extract_facts(full_text)
        facts = clean_text(facts)
        if not facts or len(facts) < min_facts_chars:
            stats["facts_not_found_or_short"] += 1
            continue

        # Prefer explicit download_label from the downloader; fallback to exact classifier.
        dl = str(row.get("download_label", "")).strip().lower()
        if dl in {"violation", "non-violation", "mixed"}:
            label_name = dl
            v_arts = extract_target_articles(row.get("violation", ""))
            nv_arts = extract_target_articles(row.get("nonviolation", ""))
        else:
            label_name, v_arts, nv_arts = classify_case(row.to_dict())

        if label_name == "mixed" and not keep_mixed:
            stats["skipped_mixed"] += 1
            continue
        if label_name not in {"violation", "non-violation"}:
            stats["out_of_scope_or_unlabeled"] += 1
            continue

        # Keep article columns consistent with notebook usage.
        matched_v = str(row.get("matched_violation_articles", "")).strip()
        matched_nv = str(row.get("matched_nonviolation_articles", "")).strip()
        v_arts_str = matched_v if matched_v else ";".join(v_arts)
        nv_arts_str = matched_nv if matched_nv else ";".join(nv_arts)

        # Binary label.
        label = 1 if label_name == "violation" else 0

        processed_data.append(
            {
                "text": facts,
                "label": label,
                "item_id": item_id,
                "respondent": normalize_respondent(row.get("respondent", "UNKNOWN")),
                "violation_articles": v_arts_str,
                "nonviolation_articles": nv_arts_str,
            }
        )
        stats["kept"] += 1

    df_proc = pd.DataFrame(processed_data)
    if df_proc.empty:
        raise RuntimeError("No valid processed rows. Check extraction/label filters.")

    # Year merge for downstream analyses.
    year_df = df_meta[["itemid", "judgementdate"]].copy()
    year_df["itemid"] = year_df["itemid"].astype(str)
    df_proc = df_proc.merge(
        year_df.rename(columns={"itemid": "item_id"}),
        on="item_id",
        how="left",
    )
    df_proc["year"] = pd.to_datetime(df_proc["judgementdate"], dayfirst=True, errors="coerce").dt.year
    df_proc = df_proc.drop(columns=["judgementdate"])

    output_dir = os.path.join(os.path.dirname(data_dir), "processed")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed.csv")
    df_proc.to_csv(out_path, index=False)

    print("=" * 72)
    print("PREPROCESS SUMMARY")
    print("=" * 72)
    for k, v in stats.items():
        print(f"{k:28s}: {v}")
    print(f"saved_processed_csv           : {out_path}")
    print("\nlabel distribution:")
    print(df_proc["label"].value_counts(dropna=False))
    print("\ncountry x label:")
    print(pd.crosstab(df_proc["respondent"], df_proc["label"]))

    return df_proc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ECHR data with exact article matching and strict label handling."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path containing metadata.csv and full_text.json",
    )
    parser.add_argument(
        "--min_facts_chars",
        type=int,
        default=100,
        help="Minimum FACTS section length to keep a row.",
    )
    parser.add_argument(
        "--keep_mixed",
        action="store_true",
        help="Keep mixed cases (not recommended for binary classification).",
    )
    args = parser.parse_args()

    preprocess(
        data_dir=args.data_dir,
        min_facts_chars=args.min_facts_chars,
        keep_mixed=args.keep_mixed,
    )
