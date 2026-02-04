
import argparse
from echr_extractor import get_echr_extra
import os
import pandas as pd
import json

def download_data(args):
    print(f"Downloading {args.count} cases...")
    df, full_texts = get_echr_extra(
        count=args.count,
        language=['ENG'],
        verbose=True,
        save_file='n' # We will handle saving manually to ensure location
    )
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    # Save full texts
    text_path = os.path.join(output_dir, "full_text.json")
    with open(text_path, 'w') as f:
        json.dump(full_texts, f, indent=2)
    print(f"Full text saved to {text_path}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Number of cases to download")
    args = parser.parse_args()
    
    download_data(args)
