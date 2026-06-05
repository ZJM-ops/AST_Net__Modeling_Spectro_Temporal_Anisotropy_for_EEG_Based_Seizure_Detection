import os
import glob
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define workspace paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
FIF_ROOT_DIR = SCRIPT_DIR / "data" / "preprocessed_data_clean1"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "data" / "segmented_data_2s"
LABELS_CSV_PATH = SCRIPT_DIR / "data" / "labels.csv"

# Segmentation parameters
WINDOW_LENGTH_SEC = 2
OVERLAP_RATIO = 0.5

# Define label mapping for consistency
LABEL_MAPPING = {
    'Baseline': 'Baseline', 'Pre-30m': 'Pre-30m', 'Pre-20m': 'Pre-20m',
    'Pre-10m': 'Pre-10m', 'Ictal': 'Ictal', 'Post-10m': 'Post-10m',
    'Post-1h': 'Post-1h', 'Post-2h': 'Post-2h', 'Post-3h': 'Post-3h',
    'Chronic-1d': 'Chronic-1d', 'Chronic-3d': 'Chronic-3d',
    'Chronic-7d': 'Chronic-7d', 'Chronic-28d': 'Chronic-28d'
}
TARGET_LABELS = list(LABEL_MAPPING.keys())

# Ensure output directory exists
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
try:
    labels_df = pd.read_csv(LABELS_CSV_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at {LABELS_CSV_PATH}")
    exit()

# Verify file system consistency
matched_metadata = []
print("Verifying consistency between CSV and .fif files...")
relevant_records = labels_df[labels_df['label'].isin(TARGET_LABELS)]

for _, row in tqdm(relevant_records.iterrows(), total=len(relevant_records), desc="File Check"):
    animal_id = str(row['animal_id']).strip()
    group = str(row['group']).strip()
    label = str(row['label']).strip()
    
    # Reconstruct expected .fif filename
    original_filename_base = Path(row['filepath']).stem
    expected_fif_name = f"{animal_id}_{original_filename_base}-raw.fif".replace(' ', '_')
    expected_fif_path = FIF_ROOT_DIR / group / label / expected_fif_name

    if expected_fif_path.exists():
        matched_metadata.append({
            'fif_filepath': expected_fif_path,
            'animal_id': animal_id,
            'group': group,
            'original_label': label
        })
    else:
        tqdm.write(f"  [Warning] Missing file: {expected_fif_path}")

if not matched_metadata:
    print("Error: No valid matching files found for segmentation.")
    exit()

print(f"\nMatched {len(matched_metadata)} files. Starting segmentation...")

processed_count = 0
skipped_count = 0

# Loop through files and perform sliding window segmentation
for meta in tqdm(matched_metadata, desc="Segmenting"):
    fif_filepath = meta['fif_filepath']
    animal_id = meta['animal_id']
    group = meta['group']
    original_label = meta['original_label']
    
    new_label = LABEL_MAPPING[original_label]
    output_dir = OUTPUT_ROOT_DIR / group / new_label
    
    # Check for existing segments to avoid redundant processing
    output_file_pattern = f"{animal_id}_{group}_{new_label}_seg*.npy"
    existing_segments = list(output_dir.glob(output_file_pattern))
    
    if existing_segments:
        skipped_count += 1
        continue
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = mne.io.read_raw_fif(str(fif_filepath), preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        window_samples = int(WINDOW_LENGTH_SEC * sfreq)
        step_samples = int(window_samples * (1 - OVERLAP_RATIO))
        
        if step_samples <= 0:
            raise ValueError("Invalid step size. Check window length and overlap ratio.")

        num_segments = 0
        total_samples = raw.n_times
        
        # Sliding window loop
        for start_sample in range(0, total_samples - window_samples + 1, step_samples):
            segment_data = raw.get_data(start=start_sample, stop=start_sample + window_samples)
            segment_filename = f"{animal_id}_{group}_{new_label}_seg{num_segments:04d}.npy"
            output_path = output_dir / segment_filename
            
            np.save(output_path, segment_data)
            num_segments += 1
        
        processed_count += 1
        
    except Exception as e:
        tqdm.write(f"  [Error] Failed to process {fif_filepath}: {e}")

# Final summary report
print("\n--- Segmentation Summary ---")
print(f"Total relevant records: {len(relevant_records)}")
print(f"Newly processed:        {processed_count}")
print(f"Skipped (already exist): {skipped_count}")
print(f"Total successful:       {processed_count + skipped_count}")

error_count = len(matched_metadata) - processed_count - skipped_count
if error_count > 0:
    print(f"Failed due to errors:   {error_count}")