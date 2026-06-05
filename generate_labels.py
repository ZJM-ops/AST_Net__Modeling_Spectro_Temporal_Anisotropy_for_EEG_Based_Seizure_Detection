import re
from pathlib import Path
import os
import pandas as pd

def create_label_file(root_folder: Path, output_filepath: Path):
    """Scan EEG files, parse metadata, and generate label mapping files."""
    parsed_data = []
    print(f"Scanning directory: {root_folder}\n")

    # Defined fine-grained classification labels based on filename keywords
    keywords = {
        'Baseline': {'建模前1d', '建模前1天', 'baseline', 'da00109a', 'da00109j', 'da0010a1'},
        'Ictal': {'止惊前10min'},
        
        'Pre-30m': {'iv级前30min', 'iv前30min', 'iv级发作前30min'},
        'Pre-20m': {'iv级前20min', '20min'},
        'Pre-10m': {'iv级前10min', 'iv前10min', 'iv级发作前10min', "iv发作前10min"},
        
        'Post-10m': {'止惊后10min', '止惊后10mim'},
        'Post-1h': {'止惊后1h'},
        'Post-2h': {'止惊后2h'},
        'Post-3h': {'止惊后3h'},

        'Chronic-1d': {'建模后1d', '建模后第1d', '建模后第1天'},
        'Chronic-3d': {'建模后3d', '建模后第3d', '建模后第3天'},
        'Chronic-7d': {'建模后7d', '建模后第7天', '建模后第7d', '7d-1'},
        'Chronic-28d': {'建模后28d', '建模后第28天', '建模后第28d', '止惊后28d'}
    }
    
    # Traverse directories to process .edf files
    for dirpath, dirnames, filenames in os.walk(str(root_folder)):
        for filename in filenames:
            if filename.endswith('.edf'):
                full_path = Path(dirpath) / filename
                relative_path = full_path.relative_to(root_folder)
                
                parts = relative_path.as_posix().split('/')
                if len(parts) < 3:
                    print(f"Warning: Invalid directory hierarchy, skipping -> {relative_path}")
                    continue
                
                group = parts[0]
                animal_folder_name = parts[1]
                
                # Extract animal ID from folder name
                animal_num_match = re.search(r'(\d+)', animal_folder_name)
                if animal_num_match:
                    animal_id = f"{group.replace('组', '')}_{animal_num_match.group(1)}"
                else:
                    animal_id = f"{group.replace('组', '')}_{animal_folder_name}"

                clean_filename = filename.lstrip('- ').strip().lower().replace('.edf', '')
                label = "To_Ignore"

                # Matching stage 1: Exact match
                is_matched = False
                for lbl, kw_set in keywords.items():
                    if clean_filename in kw_set:
                        label = lbl
                        is_matched = True
                        break
                
                # Matching stage 2: Partial match
                if not is_matched:
                    for lbl, kw_set in keywords.items():
                        if any(kw in clean_filename for kw in kw_set):
                            label = lbl
                            break
                
                parsed_data.append({
                    'filepath': full_path.as_posix(),
                    'animal_id': animal_id,
                    'group': group,
                    'label': label
                })

    if not parsed_data:
        print("Error: No .edf files found in the specified directory.")
        return

    df = pd.DataFrame(parsed_data)
    
    # Save valid filtered data
    core_df = df[df['label'] != 'To_Ignore'].copy()
    core_df = core_df.sort_values(by=['group', 'animal_id']).reset_index(drop=True)
    core_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    
    print("Process Complete!")
    print(f"Total .edf files scanned: {len(df)}")
    print(f"Filtered target files saved to: {output_filepath}")
    
    # Save unlabelled files for checking
    ignored_df = df[df['label'] == 'To_Ignore']
    if not ignored_df.empty:
        ignored_output_filepath = output_filepath.parent / 'ignored_labels.csv'
        ignored_df.to_csv(ignored_output_filepath, index=False, encoding='utf-8-sig')
        print(f"Unlabelled files saved to: {ignored_output_filepath}")

    print("\nPreview of filtered results:")
    print(core_df.head())
    print("\nLabel distribution:")
    print(core_df['label'].value_counts())
    print("\nGroup distribution:")
    print(core_df['group'].value_counts())

if __name__ == '__main__':
    # Define workspace path context using relative paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    ROOT_FOLDER_PATH = SCRIPT_DIR / "data" / "脑电图数据-已提取的10min脑电图"
    OUTPUT_CSV_PATH = SCRIPT_DIR / "data" / "labels.csv"
    
    if not ROOT_FOLDER_PATH.isdir() if hasattr(ROOT_FOLDER_PATH, 'isdir') else not ROOT_FOLDER_PATH.is_dir():
        print(f"Error: Target directory not found -> {ROOT_FOLDER_PATH}")
    else:
        create_label_file(ROOT_FOLDER_PATH, OUTPUT_CSV_PATH)