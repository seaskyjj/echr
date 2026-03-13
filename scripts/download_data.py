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
TARGET_COUNTRIES_DEFAULT = ["RUS", "TUR", "GBR"]

# Keep the same metadata-rich field set as HUDOC frontend + label fields.
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
    """Run one HUDOC query call and return parsed JSON."""
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
    """Fetch all pages for a HUDOC query."""
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
    """
    Exact article extraction:
    - match `3` but not `13`
    - allow separators like ';' and article suffixes like '3-1'
    """
    s = str(field_val).strip().lower()
    if not s or s in {"nan", "false", "0", ""}:
        return []

    found = []
    for art in target_articles:
        if re.search(rf"(?:^|;){re.escape(art)}(?:$|;|-)", s):
            found.append(art)
    return found


def classify_case(row: Dict, target_articles: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Return:
    - label: violation / non-violation / mixed / out_of_scope
    - matched_violation_articles
    - matched_nonviolation_articles
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


def build_country_query(country_code: str) -> str:
    # Mirrors HUDOC query style used in your original script.
    return (
        '(contentsitename=ECHR) AND (documentcollectionid2:"JUDGMENTS") '
        f'AND (respondent:"{country_code}") AND (languageisocode:"ENG")'
    )


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        cols = r.get("columns", {})
        if cols:
            records.append(cols)
    return pd.DataFrame(records)


def sample_per_label(
    df: pd.DataFrame,
    per_label_count: int,
    random_state: int,
    country_code: str,
) -> pd.DataFrame:
    out_parts = []
    for lbl in ["violation", "non-violation"]:
        sub = df[df["download_label"] == lbl].copy()
        if len(sub) == 0:
            print(f"  {country_code} {lbl}: 0 available")
            continue
        if len(sub) > per_label_count:
            sub = sub.sample(n=per_label_count, random_state=random_state)
        out_parts.append(sub)
        print(f"  {country_code} {lbl}: selected {len(sub)} / available {len(df[df['download_label'] == lbl])}")

    if not out_parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(out_parts, ignore_index=True)


def download_data(args):
    if (not args.skip_full_text) and download_full_text_main is None:
        raise RuntimeError(
            f"echr-extractor is required for full text download but failed to import: {IMPORT_ERROR}"
        )

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    target_articles = [a.strip() for a in args.articles.split(",") if a.strip()]
    random.seed(args.random_state)

    print("=" * 72)
    print("HUDOC DATA ACQUISITION")
    print(f"Countries: {countries}")
    print(f"Target articles: {target_articles}")
    print(f"Per-country per-label cap: {args.per_country_count}")
    print(f"Drop mixed labels: {not args.keep_mixed}")
    print("=" * 72)

    all_selected = []

    for country in countries:
        print(f"\n--- Fetching country: {country} ---")
        query = build_country_query(country)
        raw_rows = fetch_all_results(query=query, select=SELECT_FIELDS, batch_size=args.batch_size)
        raw_df = rows_to_dataframe(raw_rows)

        if raw_df.empty:
            print(f"  No rows returned for {country}.")
            continue

        # Ensure required columns exist even if HUDOC omits in edge responses.
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

        print(
            "  Label split before filtering:",
            raw_df["download_label"].value_counts(dropna=False).to_dict(),
        )

        if args.keep_mixed:
            keep_labels = ["violation", "non-violation", "mixed"]
        else:
            keep_labels = ["violation", "non-violation"]
        filtered = raw_df[raw_df["download_label"].isin(keep_labels)].copy()

        if filtered.empty:
            print(f"  No eligible rows left for {country} after filtering.")
            continue

        # Deduplicate within country by itemid first.
        before_dedup = len(filtered)
        filtered = filtered.drop_duplicates(subset=["itemid"], keep="first").copy()
        if len(filtered) != before_dedup:
            print(f"  Deduplicated within country: {before_dedup} -> {len(filtered)}")

        selected = sample_per_label(
            filtered,
            per_label_count=args.per_country_count,
            random_state=args.random_state,
            country_code=country,
        )
        all_selected.append(selected)

    if not all_selected:
        raise RuntimeError("No data selected for any country.")

    combined_df = pd.concat(all_selected, ignore_index=True)
    before_global_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["itemid"], keep="first").copy()
    if len(combined_df) != before_global_dedup:
        print(f"\nGlobal dedup across countries: {before_global_dedup} -> {len(combined_df)}")

    # Keep respondent stable for downstream scripts.
    combined_df["respondent"] = combined_df["respondent"].astype(str)

    print("\n" + "=" * 72)
    print("FINAL SELECTED METADATA COUNTS")
    print("=" * 72)
    print(pd.crosstab(combined_df["respondent"], combined_df["download_label"]))

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    combined_df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path} ({len(combined_df)} rows)")

    if args.skip_full_text:
        print("Skipped full text download (--skip_full_text).")
    else:
        print("Downloading full texts for selected cases...")
        json_list = download_full_text_main(combined_df, threads=args.threads)

        text_path = os.path.join(output_dir, "full_text.json")
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump(json_list, f, indent=2, ensure_ascii=False)
        print(f"Full text saved to {text_path} ({len(json_list)} documents)")

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ECHR metadata + full text from HUDOC with exact article filtering."
    )
    parser.add_argument(
        "--countries",
        type=str,
        default=",".join(TARGET_COUNTRIES_DEFAULT),
        help="Comma-separated respondent country codes (default: RUS,TUR,GBR)",
    )
    parser.add_argument(
        "--articles",
        type=str,
        default=",".join(TARGET_ARTICLES_DEFAULT),
        help="Comma-separated target article numbers (default: 3,5,6,8)",
    )
    parser.add_argument(
        "--per_country_count",
        type=int,
        default=305,
        help="Maximum rows to keep per country per label (violation/non-violation).",
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
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--keep_mixed",
        action="store_true",
        help="Keep mixed-label cases (both violation and non-violation target articles).",
    )
    parser.add_argument(
        "--skip_full_text",
        action="store_true",
        help="Only refresh metadata.csv and skip full_text.json download.",
    )
    # Legacy flag compatibility.
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="(Legacy) Alias of --per_country_count.",
    )

    args = parser.parse_args()
    if args.count is not None:
        args.per_country_count = args.count

    download_data(args)
