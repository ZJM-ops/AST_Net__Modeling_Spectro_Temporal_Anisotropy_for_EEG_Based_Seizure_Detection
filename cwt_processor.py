import json
import traceback
import warnings
from pathlib import Path
import concurrent.futures
from multiprocessing import cpu_count
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENTED_DATA_ROOT = SCRIPT_DIR / "data" / "segmented_data_2s"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "data" / "cwt_images_250-490_for_cnn_filtered"

TARGET_FOLDERS = ['Baseline', 'Ictal']

WAVELET_NAME = 'cmor1.5-1.0'
F_MIN = 250.0
F_MAX = 490.0
NUM_SCALES = 224
SAMPLING_RATE = 1000
DURATION = 2.0
SAMPLING_PERIOD = 1.0 / SAMPLING_RATE

try:
    central_frequency = pywt.central_frequency(WAVELET_NAME)
    scale_min = (central_frequency * SAMPLING_RATE) / F_MAX
    scale_max = (central_frequency * SAMPLING_RATE) / F_MIN
    SCALES = np.logspace(np.log10(scale_min), np.log10(scale_max), num=NUM_SCALES)
    
    FREQUENCIES = pywt.scale2frequency(WAVELET_NAME, SCALES) * SAMPLING_RATE
    TIME_POINTS = np.arange(0, DURATION, SAMPLING_PERIOD)
except Exception as e:
    print(f"Error initializing CWT scale metrics: {e}")
    exit()

def process_single_npy(args):
    """Generate and save 2D time-frequency scalograms for all valid file channels."""
    filepath, global_max_energy = args
    filepath = Path(filepath)
    try:
        parts = filepath.resolve().as_posix().split('/')
        label = parts[-2]
        group = parts[-3]
        base_filename = filepath.stem
        
        output_dir = OUTPUT_ROOT_DIR / group / label
        output_dir.mkdir(parents=True, exist_ok=True)
            
        segment_data = np.load(filepath)
        if segment_data.size == 0:
            return f"Warning: Empty matrix file {filepath.name}"

        processed_channels = 0
        for i, channel_data in enumerate(segment_data):
            output_path_png = output_dir / f"{base_filename}_ch{i+1}.png"
            
            if output_path_png.exists():
                continue 

            coefficients, _ = pywt.cwt(channel_data, SCALES, WAVELET_NAME, sampling_period=SAMPLING_PERIOD)
            cwt_matrix = np.abs(coefficients)
            
            # Render clean visualization metrics without borders
            fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            ax.imshow(cwt_matrix, cmap='jet', aspect='auto', 
                      extent=(0.0, DURATION, F_MIN, F_MAX),
                      vmin=0, vmax=global_max_energy)

            plt.savefig(output_path_png)
            plt.close(fig)
            processed_channels += 1
            
        return "processed" if processed_channels > 0 else "skipped"

    except Exception:
        return f"Error: Failed processing {filepath.name}\n{traceback.format_exc()}"

def calculate_max_energy_for_all(filepath):
    """Perform initial rapid scan to evaluate localized peak amplitude parameters."""
    try:
        segment_data = np.load(filepath)
        if segment_data.size == 0:
            return 0.0
        max_val_in_file = 0.0
        for channel_data in segment_data:
            coefficients, _ = pywt.cwt(channel_data, SCALES, WAVELET_NAME, sampling_period=SAMPLING_PERIOD)
            max_val_in_file = max(max_val_in_file, np.max(np.abs(coefficients)))
        return max_val_in_file
    except Exception:
        return 0.0

if __name__ == '__main__':
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory initialized: {OUTPUT_ROOT_DIR}")
    print(f"Filtering scope restricted to targets: {TARGET_FOLDERS}")

    np.save(OUTPUT_ROOT_DIR / 'frequencies.npy', FREQUENCIES)
    np.save(OUTPUT_ROOT_DIR / 'time_points.npy', TIME_POINTS)
    print("Saved matrix support artifacts ('frequencies.npy' / 'time_points.npy')")

    # Locate and query existing files inside data subdirectories
    all_npy_files = list(SEGMENTED_DATA_ROOT.rglob('*.npy'))
    if not all_npy_files:
        print(f"Error: Target data directory is currently empty -> {SEGMENTED_DATA_ROOT}")
        exit()
    
    print(f"\nFiltering pipeline elements... Total crawled segments: {len(all_npy_files)}")
    npy_files = [fp for fp in all_npy_files if fp.parent.name in TARGET_FOLDERS]

    if not npy_files:
        print(f"Error: Intersection yielded 0 matching elements inside {TARGET_FOLDERS}")
        exit()
    print(f"Filtered scope completed. Segments marked for generation: {len(npy_files)}")

    num_cores = cpu_count()
    max_value_file = OUTPUT_ROOT_DIR / "global_max_energy.json"
    global_max_energy = 0.0

    # Retrieve energy threshold parameter configurations from repository cache
    if max_value_file.exists():
        try:
            with open(max_value_file, 'r') as f:
                global_max_energy = json.load(f).get("global_max_energy", 0.0)
            if global_max_energy > 0:
                print(f"\nLoaded global energy peak from configuration cache: {global_max_energy:.4f}")
            else:
                global_max_energy = 0.0
        except Exception as e:
            print(f"Warning: Failed reading configuration parameters ({e}), recalculating energy maps.")
            global_max_energy = 0.0

    if global_max_energy == 0.0:
        print("\nPhase 1: Computing unified maximum energy map across filtered cohorts...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(calculate_max_energy_for_all, [str(f) for f in npy_files]), 
                                total=len(npy_files), desc="Peak Scan"))
        
        valid_results = [r for r in results if r is not None and r > 0]
        global_max_energy = np.max(valid_results) if valid_results else 0.0

        if global_max_energy > 0:
            with open(max_value_file, 'w') as f:
                json.dump({"global_max_energy": global_max_energy}, f, indent=4)
            print(f"Computed matrix scaling reference metric: {global_max_energy:.4f}")
        else:
            print("Fatal Error: Calculated global maximum map resolution returned 0.")
            exit()
            
    # Phase 2: Parallelized CWT 2D Image Mapping
    print("\nPhase 2: Generating scalogram matrix frames (PNG export)...")
    process_args = [(str(fp), global_max_energy) for fp in npy_files]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_single_npy, process_args), total=len(process_args), desc="CWT Generation"))
    
    # Execution Diagnostic Analytics
    processed_count = results.count("processed")
    skipped_count = results.count("skipped")
    error_results = [r for r in results if r and ("Error" in r or "Warning" in r)]
    
    print("\n--- Pipeline Run Summary ---")
    print(f"Newly Rendered Scalograms: {processed_count}")
    print(f"Skipped Frames (cache match): {skipped_count}")
    
    if error_results:
        print(f"Anomalies logged: {len(error_results)}")
        error_log_path = OUTPUT_ROOT_DIR / "cwt_generation_errors.log"
        with open(error_log_path, 'w', encoding='utf-8') as f:
            for res in error_results:
                f.write(str(res) + '\n')
