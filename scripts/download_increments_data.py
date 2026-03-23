import argparse
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

import pandas as pd

try:
    from echr_extractor.ECHR_html_downloader import download_full_text_main
except Exception as e:
    download_full_text_main = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


HUDOC_RESULTS_URL = "https://hudoc.echr.coe.int/app/query/results"
TARGET_ARTICLES_DEFAULT = ["3", "5", "6", "8"]
TARGET_COUNTRIES_INCREMENT_DEFAULT = ["FRA", "ROU", "POL", "DEU", "BEL"]

# Keep aligned with the existing raw metadata schema.
SELECT_FIELDS = (
    "itemid,applicability,appno,article,conclusion,decisiondate,docname,documentcollectionid,"
    "documentcollectionid2,doctype,externalsources,importance,introductiondate,issue,judgementdate,"
    "kpthesaurus,meetingnumber,originatingbody,publishedby,referencedate,kpdate,advopidentifier,advopstatus,"
    "reportdate,representedby,resolutiondate,resolutionnumber,respondent,rulesofcourt,separateopinion,scl,"
    "typedescription,ecli,casecitation,ECHRConcepts,violation,nonviolation,languageisocode,contentsitename"
)


def hudoc_query(
    query: str,
    select: str = "itemid",
    start: int = 0,
    length: int = 200,
    sort: str = "",
    retries: int = 3,
    sleep_sec: float = 1.0,
) -> Dict:
    params = {
        "query": query,
        "select": select,
        "sort": sort,
        "start": str(start),
        "length": str(length),
    }
    url = HUDOC_RESULTS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_sec * attempt)

    raise RuntimeError(f"HUDOC query failed after {retries} retries: {last_err}")


def fetch_all_results(query: str, select: str, batch_size: int = 200) -> List[Dict]:
    first = hudoc_query(query=query, select=select, start=0, length=0)
    total = int(first.get("resultcount", 0))
    if total == 0:
        return []

    results = []
    for start in range(0, total, batch_size):
        page = hudoc_query(query=query, select=select, start=start, length=batch_size)
        page_rows = page.get("results", [])
        if not page_rows:
            break
        results.extend(page_rows)
    return results


def extract_target_articles(field_val: str, target_articles: List[str]) -> List[str]:
    s = str(field_val).strip().lower()
    if not s or s in {"nan", "false", "0", ""}:
        return []

    found = []
    for art in target_articles:
        if re.search(rf"(?:^|;){re.escape(art)}(?:$|;|-)", s):
            found.append(art)
    return found


def classify_case(row: Dict, target_articles: List[str]) -> Tuple[str, List[str], List[str]]:
    v_arts = extract_target_articles(row.get("violation", ""), target_articles)
    nv_arts = extract_target_articles(row.get("nonviolation", ""), target_articles)

    if v_arts and not nv_arts:
        return "violation", v_arts, nv_arts
    if nv_arts and not v_arts:
        return "non-violation", v_arts, nv_arts
    if v_arts and nv_arts:
        return "mixed", v_arts, nv_arts
    return "out_of_scope", v_arts, nv_arts


def build_country_query(country_code: str) -> str:
    return (
        '(contentsitename=ECHR) AND (documentcollectionid2:"JUDGMENTS") '
        f'AND (respondent:"{country_code}") AND (languageisocode:"ENG")'
    )


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        cols = row.get("columns", {})
        if cols:
            records.append(cols)
    return pd.DataFrame(records)


def add_respondent_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "respondent" not in df.columns:
        df["respondent"] = ""
    respondent_str = df["respondent"].fillna("").astype(str).str.strip()
    df["is_multi_respondent"] = respondent_str.str.contains(";", regex=False)
    df["respondent_list"] = respondent_str.apply(
        lambda s: ";".join([part.strip() for part in s.split(";") if part.strip()])
    )
    return df


def sample_per_label(
    df: pd.DataFrame,
    per_label_count: int,
    random_state: int,
    country_code: str,
) -> pd.DataFrame:
    if per_label_count <= 0:
        print(
            f"  {country_code}: keeping all eligible rows "
            f"({len(df[df['download_label'].isin(['violation', 'non-violation'])])})"
        )
        return df.copy()

    out_parts = []
    for lbl in ["violation", "non-violation"]:
        sub = df[df["download_label"] == lbl].copy()
        if len(sub) == 0:
            print(f"  {country_code} {lbl}: 0 available")
            continue
        if len(sub) > per_label_count:
            sub = sub.sample(n=per_label_count, random_state=random_state)
        out_parts.append(sub)
        print(
            f"  {country_code} {lbl}: selected {len(sub)} / "
            f"available {len(df[df['download_label'] == lbl])}"
        )

    if not out_parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(out_parts, ignore_index=True)


def load_existing_metadata(existing_raw_dir: str) -> pd.DataFrame:
    metadata_path = os.path.join(existing_raw_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        return pd.DataFrame()
    return pd.read_csv(metadata_path)


def load_existing_full_text(existing_raw_dir: str) -> List[Dict]:
    text_path = os.path.join(existing_raw_dir, "full_text.json")
    if not os.path.exists(text_path):
        return []
    with open(text_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {text_path}, got {type(data).__name__}")
    return data


def merge_metadata(existing_df: pd.DataFrame, increment_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        return increment_df.copy()
    if increment_df.empty:
        return existing_df.copy()

    existing_df = existing_df.copy()
    increment_df = increment_df.copy()

    all_cols = list(existing_df.columns)
    for col in increment_df.columns:
        if col not in all_cols:
            all_cols.append(col)

    existing_df = existing_df.reindex(columns=all_cols)
    increment_df = increment_df.reindex(columns=all_cols)

    merged = pd.concat([existing_df, increment_df], ignore_index=True)
    merged["itemid"] = merged["itemid"].astype(str)
    merged = merged.drop_duplicates(subset=["itemid"], keep="first").copy()
    return merged


def merge_full_text(existing_texts: List[Dict], increment_texts: List[Dict]) -> List[Dict]:
    merged = {}
    for item in existing_texts + increment_texts:
        item_id = str(item.get("item_id", "")).strip()
        if not item_id:
            continue
        if item_id not in merged:
            merged[item_id] = item
    return list(merged.values())


def write_json(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_summary(path: str, summary: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def download_increment_data(args):
    if (not args.skip_full_text) and download_full_text_main is None:
        raise RuntimeError(
            f"echr-extractor is required for full text download but failed to import: {IMPORT_ERROR}"
        )

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    target_articles = [a.strip() for a in args.articles.split(",") if a.strip()]
    random.seed(args.random_state)

    existing_df = load_existing_metadata(args.existing_raw_dir)
    existing_itemids = set()
    if not existing_df.empty and "itemid" in existing_df.columns:
        existing_itemids = set(existing_df["itemid"].astype(str).tolist())

    print("=" * 72)
    print("HUDOC INCREMENTAL DATA ACQUISITION")
    print(f"Countries: {countries}")
    print(f"Target articles: {target_articles}")
    print(f"Per-country per-label cap: {args.per_country_count} (0 means keep all)")
    print(f"Drop mixed labels: {not args.keep_mixed}")
    print(f"Existing raw dir: {args.existing_raw_dir}")
    print(f"Existing metadata rows: {len(existing_df)}")
    print(f"Existing unique itemids: {len(existing_itemids)}")
    print("=" * 72)

    all_selected = []
    summary = {
        "countries": countries,
        "articles": target_articles,
        "existing_metadata_rows": int(len(existing_df)),
        "existing_unique_itemids": int(len(existing_itemids)),
        "per_country": {},
    }

    for country in countries:
        print(f"\n--- Fetching country: {country} ---")
        query = build_country_query(country)
        raw_rows = fetch_all_results(query=query, select=SELECT_FIELDS, batch_size=args.batch_size)
        raw_df = rows_to_dataframe(raw_rows)

        if raw_df.empty:
            print(f"  No rows returned for {country}.")
            summary["per_country"][country] = {"raw_rows": 0}
            continue

        for col in ["violation", "nonviolation", "itemid", "respondent"]:
            if col not in raw_df.columns:
                raw_df[col] = ""

        labels = []
        v_matched = []
        nv_matched = []
        for _, row in raw_df.iterrows():
            lbl, v_arts, nv_arts = classify_case(row.to_dict(), target_articles)
            labels.append(lbl)
            v_matched.append(";".join(v_arts))
            nv_matched.append(";".join(nv_arts))

        raw_df["download_label"] = labels
        raw_df["matched_violation_articles"] = v_matched
        raw_df["matched_nonviolation_articles"] = nv_matched
        raw_df["itemid"] = raw_df["itemid"].astype(str)

        print("  Label split before filtering:", raw_df["download_label"].value_counts(dropna=False).to_dict())

        if args.keep_mixed:
            keep_labels = ["violation", "non-violation", "mixed"]
        else:
            keep_labels = ["violation", "non-violation"]
        filtered = raw_df[raw_df["download_label"].isin(keep_labels)].copy()

        before_existing_filter = len(filtered)
        filtered = filtered[~filtered["itemid"].isin(existing_itemids)].copy()
        skipped_existing = before_existing_filter - len(filtered)
        if skipped_existing:
            print(f"  Skipped already-existing itemids: {skipped_existing}")

        if filtered.empty:
            print(f"  No eligible new rows left for {country} after filtering.")
            summary["per_country"][country] = {
                "raw_rows": int(len(raw_df)),
                "eligible_rows": int(before_existing_filter),
                "skipped_existing": int(skipped_existing),
                "selected_rows": 0,
            }
            continue

        before_dedup = len(filtered)
        filtered = filtered.drop_duplicates(subset=["itemid"], keep="first").copy()
        if len(filtered) != before_dedup:
            print(f"  Deduplicated within country: {before_dedup} -> {len(filtered)}")

        filtered = add_respondent_flags(filtered)
        filtered["source_batch"] = args.batch_name
        filtered["source_type"] = "increment"

        selected = sample_per_label(
            filtered,
            per_label_count=args.per_country_count,
            random_state=args.random_state,
            country_code=country,
        )

        summary["per_country"][country] = {
            "raw_rows": int(len(raw_df)),
            "eligible_rows": int(before_existing_filter),
            "skipped_existing": int(skipped_existing),
            "selected_rows": int(len(selected)),
            "selected_label_split": {
                k: int(v) for k, v in selected["download_label"].value_counts(dropna=False).to_dict().items()
            },
        }

        if not selected.empty:
            all_selected.append(selected)

    if not all_selected:
        raise RuntimeError("No new data selected for any country.")

    increment_df = pd.concat(all_selected, ignore_index=True)
    before_global_dedup = len(increment_df)
    increment_df = increment_df.drop_duplicates(subset=["itemid"], keep="first").copy()
    if len(increment_df) != before_global_dedup:
        print(f"\nGlobal dedup across countries: {before_global_dedup} -> {len(increment_df)}")

    os.makedirs(args.output_dir, exist_ok=True)

    increment_metadata_path = os.path.join(args.output_dir, "metadata_increment.csv")
    increment_df.to_csv(increment_metadata_path, index=False)
    print(f"\nIncrement metadata saved to {increment_metadata_path} ({len(increment_df)} rows)")
    print(pd.crosstab(increment_df["respondent"], increment_df["download_label"]))

    increment_texts = []
    if args.skip_full_text:
        print("Skipped full text download (--skip_full_text).")
    else:
        print("Downloading full texts for increment cases...")
        increment_texts = download_full_text_main(increment_df, threads=args.threads)
        increment_text_path = os.path.join(args.output_dir, "full_text_increment.json")
        write_json(increment_text_path, increment_texts)
        print(f"Increment full text saved to {increment_text_path} ({len(increment_texts)} documents)")

    existing_texts = load_existing_full_text(args.existing_raw_dir)
    merged_metadata_df = merge_metadata(existing_df, increment_df)
    merged_metadata_df = add_respondent_flags(merged_metadata_df)
    merged_metadata_path = os.path.join(args.output_dir, "metadata_merged.csv")
    merged_metadata_df.to_csv(merged_metadata_path, index=False)
    print(f"Merged metadata saved to {merged_metadata_path} ({len(merged_metadata_df)} rows)")

    if args.skip_full_text:
        merged_texts = existing_texts
    else:
        merged_texts = merge_full_text(existing_texts, increment_texts)
        merged_text_path = os.path.join(args.output_dir, "full_text_merged.json")
        write_json(merged_text_path, merged_texts)
        print(f"Merged full text saved to {merged_text_path} ({len(merged_texts)} documents)")

    summary.update(
        {
            "batch_name": args.batch_name,
            "increment_metadata_rows": int(len(increment_df)),
            "merged_metadata_rows": int(len(merged_metadata_df)),
            "existing_full_text_rows": int(len(existing_texts)),
            "increment_full_text_rows": int(len(increment_texts)),
            "merged_full_text_rows": int(len(merged_texts)),
            "increment_country_label_table": pd.crosstab(
                increment_df["respondent"], increment_df["download_label"]
            ).to_dict(),
        }
    )
    summary_path = os.path.join(args.output_dir, "increment_summary.json")
    write_summary(summary_path, summary)
    print(f"Summary saved to {summary_path}")

    if args.write_merged_to_base:
        base_metadata_path = os.path.join(args.existing_raw_dir, "metadata.csv")
        merged_metadata_df.to_csv(base_metadata_path, index=False)
        print(f"Overwrote base metadata with merged metadata: {base_metadata_path}")

        if not args.skip_full_text:
            base_text_path = os.path.join(args.existing_raw_dir, "full_text.json")
            write_json(base_text_path, merged_texts)
            print(f"Overwrote base full text with merged full text: {base_text_path}")

    return increment_df, merged_metadata_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download ECHR incremental metadata + full text from HUDOC and produce both "
            "increment-only and merged outputs compatible with the existing dataset."
        )
    )
    parser.add_argument(
        "--countries",
        type=str,
        default=",".join(TARGET_COUNTRIES_INCREMENT_DEFAULT),
        help="Comma-separated respondent country codes for the next increment wave.",
    )
    parser.add_argument(
        "--articles",
        type=str,
        default=",".join(TARGET_ARTICLES_DEFAULT),
        help="Comma-separated target article numbers (default: 3,5,6,8).",
    )
    parser.add_argument(
        "--existing_raw_dir",
        type=str,
        default="data/raw",
        help="Existing raw directory containing base metadata.csv and full_text.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/incremental/raw",
        help="Output directory for metadata_increment/full_text_increment and merged files.",
    )
    parser.add_argument(
        "--per_country_count",
        type=int,
        default=0,
        help="Maximum rows to keep per country per label (0 means keep all eligible rows).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="HUDOC pagination batch size.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of threads for full text download.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling when caps are used.",
    )
    parser.add_argument(
        "--batch_name",
        type=str,
        default="increment_fra_rou_pol_deu_bel",
        help="Source batch tag to store in the increment metadata.",
    )
    parser.add_argument(
        "--keep_mixed",
        action="store_true",
        help="Keep mixed-label cases (both violation and non-violation target articles).",
    )
    parser.add_argument(
        "--skip_full_text",
        action="store_true",
        help="Only refresh metadata_increment/metadata_merged and skip full text download.",
    )
    parser.add_argument(
        "--write_merged_to_base",
        action="store_true",
        help="Overwrite data/raw/metadata.csv and full_text.json with merged outputs.",
    )

    args = parser.parse_args()
    download_increment_data(args)
