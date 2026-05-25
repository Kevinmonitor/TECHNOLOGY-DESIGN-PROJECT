import os
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Basic text cleaning matching your exploratory framework
    return text.strip().lower()

def run_pipeline():
    print("[DE PIPELINE] Starting production text-processing pipeline...")
    
    raw_fpb_path = "data/raw/fpb_5class.csv"
    if not os.path.exists(raw_fpb_path):
        raw_fpb_path = "../data/raw/fpb_5class.csv"
        
    if not os.path.exists(raw_fpb_path):
        print(f"Error: Could not find base dataset at {raw_fpb_path}")
        return

    df = pd.read_csv(raw_fpb_path)
    print(f"[DE PIPELINE] Loaded base dataset with {len(df)} rows.")
    
    # Run structural engineering steps
    df['text'] = df['text'].apply(clean_text)
    
    # Generate strict stratified 70/15/15 splits to protect against data leakage
    print("[DE PIPELINE] Splitting datasets symmetrically...")
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df['label'])
    
    # Export to our single production source of truth directory
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    
    train_df.to_csv("data/processed/fpb_train.csv", index=False)
    val_df.to_csv("data/processed/fpb_val.csv", index=False)
    test_df.to_csv("data/processed/fpb_test.csv", index=False)
    
    print(f"[DE PIPELINE] Completed successfully! Saved splits to data/processed/")
    print(f"       -> Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

if __name__ == "__main__":
    run_pipeline()
