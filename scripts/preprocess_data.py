
import os
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_facts(text):
    # Simple heuristic to extract FACTS section
    # Valid sections often start with "THE FACTS" or "AS TO THE FACTS"
    # And end with "THE LAW" or "PROCEEDINGS BEFORE THE COMMISSION"
    
    normalization_map = {
        "AS TO THE FACTS": "THE FACTS",
        "THE FACTS": "THE FACTS"
    }
    
    text_upper = text.upper()
    
    start_idx = -1
    for key in normalization_map:
        idx = text_upper.find(key)
        if idx != -1:
            start_idx = idx + len(key)
            break
            
    if start_idx == -1:
        return None
        
    # Find end
    end_markers = ["THE LAW", "PROCEEDINGS BEFORE THE COMMISSION", "RELEVANT DOMESTIC LAW"]
    end_idx = -1
    
    current_search_start = start_idx
    best_end_idx = len(text)
    
    for marker in end_markers:
        idx = text_upper.find(marker, current_search_start)
        if idx != -1 and idx < best_end_idx:
            best_end_idx = idx
            
    return text[start_idx:best_end_idx].strip()

def preprocess(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    full_text_path = os.path.join(data_dir, "full_text.json")
    
    if not os.path.exists(metadata_path) or not os.path.exists(full_text_path):
        print("Data files not found.")
        return

    df_meta = pd.read_csv(metadata_path)
    with open(full_text_path, 'r') as f:
        full_texts = json.load(f)
        
    # Convert full_texts list to dict keyed by item_id
    text_map = {item['item_id']: item['full_text'] for item in full_texts}
    
    processed_data = []
    
    label_count = 0
    for _, row in df_meta.iterrows():
        item_id = row['itemid']
        if item_id not in text_map:
            # print(f"Skipping {item_id}: No text JSON entry")
            continue
            
        text = text_map[item_id]
        if not text:
            # print(f"Skipping {item_id}: Empty text")
            continue
            
        facts = extract_facts(text)
        if not facts or len(facts) < 100: # Filter simplistic/empty extractions
            # print(f"Skipping {item_id}: No FACTS section found")
            continue
            
        # Determine label
        # Violation if 'violation' column is not empty/null/FALSE
        # No Violation if 'nonviolation' is not empty/null/FALSE
        # Ignore if ambiguous or neither (e.g. inadmissable without merit judgment)
        
        is_violation = str(row['violation']).strip().lower()
        is_nonviolation = str(row['nonviolation']).strip().lower()
        
        # Simple binary mapping suitable for "Violation vs No-Violation" task
        # We focus on cases where there is a clear judgment
        label = -1
        if is_violation not in ['nan', 'false', ''] and is_violation != '0':
             label = 1
        elif is_nonviolation not in ['nan', 'false', ''] and is_nonviolation != '0':
             label = 0
        
        if label == -1:
            # print(f"Skipping {item_id}: No label (v='{row['violation']}', nv='{row['nonviolation']}')")
            continue
             
        if label != -1:
            label_count += 1
            processed_data.append({
                'text': clean_text(facts),
                'label': label,
                'item_id': item_id
            })
            
    df_proc = pd.DataFrame(processed_data)
    print(f"Processed {len(df_proc)} valid cases out of {len(df_meta)} total. Found {label_count} labeled text-having cases.")
    
    if len(df_proc) == 0:
        print("No valid cases found. Check text extraction or labeling logic.")
        return

    # Split
    train, temp = train_test_split(df_proc, test_size=0.3, random_state=42, stratify=df_proc['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    
    output_dir = os.path.join(os.path.dirname(data_dir), "processed")
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Saved processed datasets to {output_dir}")

if __name__ == "__main__":
    preprocess("data/raw")
