import os
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from pathlib import Path

# ================= CONFIGURATION =================
# 1. INPUT: Path to your trial preprocessed data
DATA_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\MCI_conn_preprocessed_full"

# 2. OUTPUT: Where to save the extracted time series
OUTPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Timeseries_Output"

# 3. ATLAS SEARCH PATH: Where we look for the atlas
ATLAS_SEARCH_DIR = r"C:\Users\DELL\nilearn_data\aal_SPM12"
# =================================================

print("--- Starting ROI Time Series Extraction (Auto-Find Mode) ---")

# --- 1. Auto-Find the Atlas File ---
print(f"Searching for 'ROI_MNI_V4.nii' in {ATLAS_SEARCH_DIR}...")

# Recursively search for the file
found_files = list(Path(ATLAS_SEARCH_DIR).rglob("ROI_MNI_V4.nii"))

if not found_files:
    print("CRITICAL ERROR: Could not find 'ROI_MNI_V4.nii'.")
    print("Please check if the 'unzip_atlas.py' script finished successfully.")
    exit()

# Take the first one found
ATLAS_FILE = str(found_files[0])
print(f"Atlas found: {ATLAS_FILE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Initialize the Masker ---
print("Initializing Masker...")
masker = NiftiLabelsMasker(labels_img=ATLAS_FILE, 
                           standardize=True, 
                           detrend=True,
                           memory='nilearn_cache', 
                           verbose=0) # reduced verbosity for cleaner output

# --- 3. Find 'swau' files ---
print(f"Scanning {DATA_DIR} for 'swau...bold.nii' files...")
func_files = list(Path(DATA_DIR).rglob("swau*_bold.nii"))

if not func_files:
    print(f"No 'swau' files found in {DATA_DIR}!")
    exit()

print(f"Found {len(func_files)} functional files. Processing now...\n")

# --- 4. Process Each Subject ---
success_count = 0

for func_file in func_files:
    subject_id = func_file.name.split('_')[0] 
    print(f"Processing: {subject_id} ...")
    
    try:
        time_series = masker.fit_transform(str(func_file))
        
        output_filename = os.path.join(OUTPUT_DIR, f"{subject_id}_timeseries.npy")
        np.save(output_filename, time_series)
        
        print(f"  -> Saved shape {time_series.shape} to: {output_filename}")
        success_count += 1
        
    except Exception as e:
        print(f" ERROR extracting {subject_id}: {e}")

print("\n" + "="*50)
print(f"DONE! Successfully processed {success_count} / {len(func_files)} subjects.")
print("="*50)