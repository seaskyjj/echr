
import argparse
from echr_extractor import get_echr_extra, get_echr
from echr_extractor.ECHR_html_downloader import download_full_text_main
import os
import pandas as pd
import json

def download_data(args):
    print(f"Targeting {args.count} cases per class (Total {args.count * 2})...")
    
    # Define queries
    # Note: HUDOC keywords for conclusion
    # We use a broad text search on conclusion field which is reliable enough
    # Use wildcard to find presence of field
    # (violation:*) matches records where violation field is not null
    query_v = '(contentsitename=ECHR) AND (documentcollectionid2:"JUDGMENTS") AND (violation:*)'
    query_nv = '(contentsitename=ECHR) AND (documentcollectionid2:"JUDGMENTS") AND (nonviolation:*)'
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    def fetch_batch(query, label_name):
        print(f"--- Fetching {label_name} candidates ---")
        # Fetch 2x count to filter for purity
        df = get_echr(
            count=args.count * 2,
            language=['ENG'],
            verbose=True,
            query_payload=query,
            save_file='n'
        )
        return df

    # 1. Fetch Metadata
    df_v = fetch_batch(query_v, "VIOLATION")
    df_nv = fetch_batch(query_nv, "NON-VIOLATION")
    
    if df_v is False: df_v = pd.DataFrame()
    if df_nv is False: df_nv = pd.DataFrame()
    
    print(f"Retrieved {len(df_v)} violation candidates and {len(df_nv)} non-violation candidates.")

    # 2. Filter for Purity (Simple Binary Classification)
    # Violation Class: Must have a violation. Ideally no 'nonviolation' to be clear, but mixed is technically a violation found.
    # Non-Violation Class: Must have nonviolation. Must NOT have violation (otherwise it's mixed/violation).
    
    def is_pure_v(row):
        v = str(row.get('violation', 'nan')).lower()
        return v not in ['nan', 'false', '', '0']

    def is_pure_nv(row):
        nv = str(row.get('nonviolation', 'nan')).lower()
        v = str(row.get('violation', 'nan')).lower()
        has_nv = nv not in ['nan', 'false', '', '0']
        has_v = v not in ['nan', 'false', '', '0']
        return has_nv and not has_v

    # Filter V
    df_v_filtered = df_v[df_v.apply(is_pure_v, axis=1)].copy()
    df_v_filtered['download_label'] = 'violation'
    df_v_final = df_v_filtered.head(args.count)
    
    # Filter NV
    df_nv_filtered = df_nv[df_nv.apply(is_pure_nv, axis=1)].copy()
    df_nv_filtered['download_label'] = 'non-violation'
    df_nv_final = df_nv_filtered.head(args.count)
    
    print(f"Filtered to {len(df_v_final)} Pure Violations and {len(df_nv_final)} Pure Non-Violations.")
    
    if len(df_v_final) < args.count or len(df_nv_final) < args.count:
        print("WARNING: Could not find enough pure cases. Proceeding with available.")

    combined_df = pd.concat([df_v_final, df_nv_final], ignore_index=True)
    combined_df.drop_duplicates(subset=['itemid'], inplace=True)
    
    if len(combined_df) == 0:
        print("No cases found.")
        return

    # 3. Download Full Text
    print("Downloading full texts for combined dataset...")
    # download_full_text_main takes a dataframe and threads
    json_list = download_full_text_main(combined_df, threads=10)
    
    # 4. Save
    metadata_path = os.path.join(output_dir, "metadata.csv")
    combined_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    text_path = os.path.join(output_dir, "full_text.json")
    with open(text_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"Full text saved to {text_path}")
    
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="Number of cases per class to download")
    args = parser.parse_args()
    
    download_data(args)
