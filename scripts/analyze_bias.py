
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path

# Suppress HF warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def resolve_split_dir(args) -> Path:
    if args.split_dir:
        return Path(args.split_dir)
    return Path(args.dataset_dir) / "data" / "processed"


def resolve_raw_dir(args) -> Path:
    if args.raw_dir:
        return Path(args.raw_dir)
    return Path(args.dataset_dir) / "data" / "raw"


def resolve_model_path(args) -> Path:
    if args.model_path:
        return Path(args.model_path)
    return Path(args.working_dir) / "results" / "legal_bert" / "final_model"


def ensure_multi_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    respondent = df.get("respondent", pd.Series([""] * len(df))).fillna("").astype(str)
    if "is_multi_respondent" not in df.columns:
        df["is_multi_respondent"] = respondent.str.contains(";", regex=False)
    else:
        df["is_multi_respondent"] = df["is_multi_respondent"].fillna(False).astype(bool)
    if "respondent_list" not in df.columns:
        df["respondent_list"] = respondent.apply(
            lambda s: ";".join([part.strip() for part in s.split(";") if part.strip()])
        )
    return df


def analyze_bias(split_dir, raw_dir, model_path, exclude_multi_respondent=True):
    print(f"Loading split data from {split_dir}, metadata from {raw_dir}, and model from {model_path}...")
    
    # Load Test Data
    test_path = os.path.join(split_dir, "test.csv")
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return
    test_df = pd.read_csv(test_path)
    
    # Load Metadata for rich features
    meta_path = os.path.join(raw_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return
    meta_df = pd.read_csv(meta_path)
    
    # Merge
    # Ensure item_id is string/int consistent
    test_df['item_id'] = test_df['item_id'].astype(str)
    meta_df['itemid'] = meta_df['itemid'].astype(str)
    
    # We only care about metadata for the test set
    merged_df = test_df.merge(meta_df, left_on='item_id', right_on='itemid', how='left')
    merged_df = ensure_multi_flags(merged_df)
    
    print(f"Loaded {len(merged_df)} test cases. Merged with metadata.")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.to(device)
    model.eval()
    
    # Predict
    predictions = []
    
    print("Running inference on test set...")
    # Batch processing would be faster but for ~80 items loop is fine and safer for memory on CPU
    for text in tqdm(merged_df['text']):
        inputs = tokenizer(
            str(text), 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
            
    merged_df['prediction'] = predictions
    merged_df['correct'] = (merged_df['prediction'] == merged_df['label'])
    
    # --- ANALYSIS ---
    
    # 1. Respondent Bias
    print("\n" + "="*30)
    print("RESPONDENT BIAS ANALYSIS")
    print("="*30)
    if 'respondent' in merged_df.columns:
        country_df = merged_df.copy()
        if exclude_multi_respondent:
            country_df = country_df[~country_df['is_multi_respondent']].copy()
            print(f"Excluded multi-respondent rows from main country analysis: {len(merged_df) - len(country_df)}")

        respondent_stats = country_df.groupby('respondent').agg(
            count=('item_id', 'count'),
            accuracy=('correct', 'mean'),
            pred_violation_rate=('prediction', 'mean'),
            actual_violation_rate=('label', 'mean')
        ).sort_values('count', ascending=False)
        
        print("\nStats by Respondent State (Top 10 by count):")
        print(respondent_stats.head(10).to_string())
        
        # Check for extreme deviation
        # e.g. Country has 0% or 100% violation rate predicted vs actual
    else:
        print("No 'respondent' column found in metadata.")

    # 2. Temporal Bias
    print("\n" + "="*30)
    print("TEMPORAL BIAS ANALYSIS")
    print("="*30)
    if 'judgementdate' in merged_df.columns:
        merged_df['year'] = pd.to_datetime(merged_df['judgementdate'], errors='coerce').dt.year
        year_stats = merged_df.groupby('year').agg(
            count=('item_id', 'count'),
            accuracy=('correct', 'mean'),
            pred_violation_rate=('prediction', 'mean'),
            actual_violation_rate=('label', 'mean')
        ).sort_values('year')
        
        print("\nStats by Judgment Year:")
        print(year_stats.to_string())
        
        # Simple correlation
        valid_years = merged_df.dropna(subset=['year'])
        if len(valid_years['prediction'].unique()) > 1:
            corr = valid_years['year'].corr(valid_years['prediction'])
            print(f"\nCorrelation (Year vs Predicted Violation): {corr:.4f}")
        else:
            print("\nCorrelation (Year vs Predicted Violation): N/A (No prediction variance)")
    else:
        print("No 'judgementdate' column found.")

    # 3. Length Bias
    print("\n" + "="*30)
    print("TEXT LENGTH BIAS ANALYSIS")
    print("="*30)
    merged_df['text_len'] = merged_df['text'].apply(lambda x: len(str(x)))
    
    # Correlation
    if len(merged_df['prediction'].unique()) > 1:
        corr_pred = merged_df['text_len'].corr(merged_df['prediction'])
        print(f"Correlation (Text Length vs Predicted Violation): {corr_pred:.4f}")
    else:
         print("Correlation (Text Length vs Predicted Violation): N/A (No prediction variance)")
         
    if len(merged_df['correct'].unique()) > 1:
        corr_acc = merged_df['text_len'].corr(merged_df['correct'])
        print(f"Correlation (Text Length vs Accuracy): {corr_acc:.4f}")
    else:
        print("Correlation (Text Length vs Accuracy): N/A (Constant Accuracy)")
    
    len_v = merged_df[merged_df['prediction']==1]['text_len'].mean()
    len_nv = merged_df[merged_df['prediction']==0]['text_len'].mean()
    print(f"\nAvg Text Length (Predicted Violation): {len_v:.1f} chars")
    if pd.notna(len_nv):
        print(f"Avg Text Length (Predicted Non-Violation): {len_nv:.1f} chars")
    else:
        print("Avg Text Length (Predicted Non-Violation): N/A (None predicted)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=".", help="Dataset root containing data/raw and data/processed.")
    parser.add_argument("--working_dir", type=str, default=".", help="Writable root for outputs and trained models.")
    parser.add_argument("--split_dir", type=str, default=None, help="Override split directory containing test.csv.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Override raw directory containing metadata.csv.")
    parser.add_argument("--model_path", type=str, default=None, help="Override trained model path.")
    parser.add_argument(
        "--include_multi_respondent",
        action="store_true",
        help="Include co-respondent cases in country-level bias analysis. Default excludes them.",
    )
    args = parser.parse_args()

    analyze_bias(
        str(resolve_split_dir(args)),
        str(resolve_raw_dir(args)),
        str(resolve_model_path(args)),
        exclude_multi_respondent=not args.include_multi_respondent,
    )
