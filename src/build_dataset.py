import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV = PROJECT_ROOT / "dataset_master.csv"

# Taxonomy Mapping Dictionary
TAXONOMY_MAP = {
    "Black cardamom(Badi Elaichi) 38": ("black_cardamom", "cardamom", "badi_elaichi", "whole_pod"),
    "black pepper 63": ("black_pepper", "pepper", "standard", "whole_seed"),
    "Cumin 58": ("cumin", "cumin", "jeera", "whole_seed"),
    "Dagad Phool (Black Stone Flower) 28": ("dagad_phool", "lichen", "standard", "dried_flower"),
    "Dalchini 26": ("dalchini", "cinnamon", "cassia", "bark"),
    "Dalchini(1) 61": ("dalchini", "cinnamon", "cassia", "bark"),
    "Dalchini(1)": ("dalchini", "cinnamon", "cassia", "bark"),
    "Dhaniya(Coriander Seeds) 13": ("dhaniya", "coriander", "standard", "whole_seed"),
    "Dried Coconut 21": ("dried_coconut", "coconut", "standard", "pieces"),
    "Dry red chillies(Byadgi or Bedgi) 69": ("chilli_byadgi", "dry_red_chilli", "byadgi", "whole_pod"),
    "Dry red chillies(Guntur Sannam) 86": ("chilli_guntur", "dry_red_chilli", "guntur_sannam", "whole_pod"),
    "Dry red chillies(Lavangi) 63": ("chilli_lavangi", "dry_red_chilli", "lavangi", "whole_pod"),
    "Dry red chillies(Sankeshwari) 74": ("chilli_sankeshwari", "dry_red_chilli", "sankeshwari", "whole_pod"),
    "Elaichi 38": ("elaichi_pod", "cardamom", "green_cardamom", "whole_pod"),
    "Elaichi Dana(Cardamom Seeds) 38": ("elaichi_dana", "cardamom", "green_cardamom", "loose_seeds"),
    "Halkund 35": ("halkund", "turmeric", "whole_rhizome", "root")
}

def validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def build_dataset():
    records = []
    
    if not RAW_DATA_DIR.exists():
        print(f"Directory {RAW_DATA_DIR} does not exist.")
        return

    print("Indexing and validating raw dataset...")
    for folder_name in os.listdir(RAW_DATA_DIR):
        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.is_dir():
            continue
            
        if folder_name not in TAXONOMY_MAP:
            print(f"Warning: Folder '{folder_name}' not in taxonomy map. Skipping.")
            continue
            
        class_label, category, sub_category, form = TAXONOMY_MAP[folder_name]
        
        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
            file_path = folder_path / file_name
            if not file_path.is_file():
                continue
                
            if validate_image(file_path):
                records.append({
                    "file_path": str(file_path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                    "folder_name": folder_name,
                    "class_label": class_label,
                    "category": category,
                    "sub_category": sub_category,
                    "form": form
                })
            else:
                print(f"Corrupted or invalid image: {file_path}")
                
    df = pd.DataFrame(records)
    
    if df.empty:
        print("No valid images found.")
        return
        
    print(f"\nTotal valid images: {len(df)}")
    
    # Two-stage stratified split: 70% train, 15% val, 15% test
    # Stage 1: split into train (70%) and temp (30%)
    try:
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.30, 
            random_state=42, 
            stratify=df['class_label']
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed for Stage 1 ({e}). Falling back to non-stratified split.")
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.30, 
            random_state=42
        )
    
    # Stage 2: split temp into val (50% of 30% = 15%) and test (50% of 30% = 15%)
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            stratify=temp_df['class_label']
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed for Stage 2 ({e}). Falling back to non-stratified split.")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42
        )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset saved to {OUTPUT_CSV}")
    
    print("\nDataset Distribution:")
    cross_tab = pd.crosstab(final_df['class_label'], final_df['split'], margins=True)
    print(cross_tab)

if __name__ == "__main__":
    build_dataset()
