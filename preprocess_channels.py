import os
from pathlib import Path
import mne
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
LABELS_CSV_PATH = SCRIPT_DIR / "data" / "labels.csv"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "data" / "preprocessed_data_clean1"
FAILED_CSV_PATH = SCRIPT_DIR / "data" / "failed_files_report.csv"

try:
    labels_df = pd.read_csv(LABELS_CSV_PATH)
except FileNotFoundError:
    print(f"Error: labels.csv not found at {LABELS_CSV_PATH}")
    exit()

required_cols = ['filepath', 'animal_id', 'group', 'label']
if not all(col in labels_df.columns for col in required_cols):
    print(f"Error: CSV file must contain columns: {required_cols}")
    exit()

# Define standard and prefixed 8-channel combinations
CHANNELS_WITH_PREFIX = [
    'EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 
    'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref'
]
CHANNELS_WITHOUT_PREFIX = [
    'Fp1-Ref', 'Fp2-Ref', 'F3-Ref', 'F4-Ref', 
    'C3-Ref', 'C4-Ref', 'P3-Ref', 'P4-Ref'
]

if not OUTPUT_ROOT_DIR.exists():
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created root output directory: {OUTPUT_ROOT_DIR}")

failed_files = []

print(f"\nStarting preprocessing for {len(labels_df)} files...")

for index, row in labels_df.iterrows():
    filepath = Path(row['filepath'])
    animal_id = row['animal_id']
    group = row['group']
    label = row['label']
    original_filename = filepath.name
    
    print(f"Processing: {group}/{label}/{original_filename}")

    try:
        if not filepath.exists():
            raise FileNotFoundError("Source file path does not exist.")
            
        target_dir = OUTPUT_ROOT_DIR / group / label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        new_filename = f"{animal_id}_{original_filename}".replace('.edf', '-raw.fif').replace(' ', '_')
        output_filepath = target_dir / new_filename
        
        if output_filepath.exists():
            print("  [Skip] Preprocessed file already exists.")
            continue

        # Inspect headers to determine channel naming format
        raw_info = mne.io.read_raw_edf(str(filepath), preload=False, encoding='latin1', verbose=False)
        
        if CHANNELS_WITH_PREFIX[0] in raw_info.ch_names:
            channels_to_use = CHANNELS_WITH_PREFIX
            print("  [Info] Prefix format 'EEG ' detected.")
        else:
            channels_to_use = CHANNELS_WITHOUT_PREFIX
            print("  [Info] Standard format without prefix detected.")
        
        # Load data and filter target 8 channels
        raw = raw_info.load_data(verbose=False)
        current_channels_to_keep = [ch for ch in channels_to_use if ch in raw.ch_names]
        
        if not current_channels_to_keep:
            raise ValueError("No matching target channels found in file.")
            
        raw.pick_channels(current_channels_to_keep)

        # Apply signal processing filters
        raw.notch_filter(freqs=range(50, 500, 50), verbose=False)
        raw.save(str(output_filepath), overwrite=True)
        print(f"  [Success] Saved to -> {output_filepath}")

    except Exception as e:
        error_message = str(e).replace('\n', ' ').strip()
        print(f"  [Error] {error_message}")
        failed_files.append({
            'filepath': str(filepath), 
            'animal_id': animal_id, 
            'group': group, 
            'label': label, 
            'reason': error_message
        })

print("\nAll preprocessing tasks finished.")

# Export execution logs if anomalies detected
if failed_files:
    print("\nAnomalies detected in the following files:")
    failed_df = pd.DataFrame(failed_files)
    for i, fail_row in failed_df.iterrows():
        print(f"  - File: {fail_row['filepath']} (ID: {fail_row['animal_id']})")
        print(f"    Reason: {fail_row['reason']}")
    failed_df.to_csv(FAILED_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\nDetailed failure logs saved to: {FAILED_CSV_PATH}")
else:
    print("\nAll tasks executed successfully without errors.")