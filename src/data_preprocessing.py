import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Production 7-step Text Normalisation"""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)           # Remove URLs
    text = re.sub(r'@\w+', '', text)                         # Remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)                   # Strip hashtags
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)             # Remove non-ASCII
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = re.sub(r'\d+', 'NUM', text)                       # Replace numbers
    return re.sub(r'\s+', ' ', text).strip()                 # Normalise whitespace

def run_pipeline():
    print("[DE PIPELINE] Upgrading datasets with advanced NLP normalisation...")
    
    # Load raw datasets
    raw_fpb = "data/raw/fpb_5class.csv"
    raw_go = "data/raw/goemotions_5class.csv"
    if not os.path.exists(raw_fpb):
        raw_fpb, raw_go = "../" + raw_fpb, "../" + raw_go

    df_fpb = pd.read_csv(raw_fpb)
    df_go = pd.read_csv(raw_go)
    
    print("[DE PIPELINE] Baking 'clean_text' directly into CSVs...")
    df_fpb['clean_text'] = df_fpb['text'].apply(clean_text)
    df_go['clean_text'] = df_go['text'].apply(clean_text)
    
    print("[DE PIPELINE] Splitting FPB dataset symmetrically...")
    train_df, test_df = train_test_split(df_fpb, test_size=0.30, random_state=42, stratify=df_fpb['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df['label'])
    
    # Export all files to the processed directory
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/fpb_train.csv", index=False)
    val_df.to_csv("data/processed/fpb_val.csv", index=False)
    test_df.to_csv("data/processed/fpb_test.csv", index=False)
    df_go.to_csv("data/processed/goemotions_5class.csv", index=False)
    
    print("[DE PIPELINE] Success! Ready for ML consumption.")

if __name__ == "__main__":
    run_pipeline()
