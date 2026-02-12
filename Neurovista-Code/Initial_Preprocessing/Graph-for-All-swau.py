import os
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image
from pathlib import Path
import warnings
import nibabel as nib

warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
# 1. INPUT: Path where you copied all your swau files
DATA_DIR = r"C:\Users\DELL\Desktop\4th-year-project\all_swau_preprocessed"

# 2. OUTPUT: Specific folder for MCI results
OUTPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\Timeseries_Output_MCI"

# 3. ATLAS SEARCH PATH: Your specific AAL path
ATLAS_SEARCH_DIR = r"C:\Users\DELL\nilearn_data\aal_SPM12"

# 4. PREPROCESSING PARAMETERS
LOW_PASS = 0.1   
HIGH_PASS = 0.01 
DEFAULT_TR = 3.0 # Fallback if header TR is missing

# 5. QUALITY CONTROL
MIN_TIMEPOINTS = 50    
MAX_TIMEPOINTS = 1000  
MIN_NONZERO_VOXELS = 1000  

# 6. MASKER SETTINGS
STANDARDIZE = 'zscore_sample'  
DETREND = True                 
SMOOTHING_FWHM = None          
# =================================================

print("="*70)
print("MCI PROJECT: ENHANCED ROI TIME SERIES EXTRACTION")
print("="*70)

# --- 1. Auto-Find the Atlas File ---
print(f"\nSearching for AAL atlas in: {ATLAS_SEARCH_DIR}")
found_files = list(Path(ATLAS_SEARCH_DIR).rglob("ROI_MNI_V4.nii"))

if not found_files:
    print("❌ CRITICAL ERROR: Could not find 'ROI_MNI_V4.nii'.")
    exit()

ATLAS_FILE = str(found_files[0])
print(f"✅ Atlas found: {ATLAS_FILE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Find Functional Files ---
print(f"\nScanning {DATA_DIR} for 'swau*.nii' files...")
func_files = list(Path(DATA_DIR).rglob("swau*.nii"))

if not func_files:
    print(f"❌ No 'swau' files found in {DATA_DIR}!")
    exit()

print(f"Found {len(func_files)} MCI functional files.\n")

# --- 3. Process Each Subject ---
success_count = 0
failed_subjects = []

for idx, func_file in enumerate(func_files, 1):
    filename = func_file.stem
    subject_id = filename.replace('swau', '').replace('_bold', '')
    
    try:
        # Load image to check header TR
        func_img = image.load_img(str(func_file))
        header_tr = func_img.header.get_zooms()[3]
        
        # Use header TR if valid, else use default
        current_tr = header_tr if header_tr > 0 else DEFAULT_TR
        
        print(f"[{idx}/{len(func_files)}] Processing: {subject_id} (TR: {current_tr}s)")
        
        # Initialize Masker for this specific TR
        masker = NiftiLabelsMasker(
            labels_img=ATLAS_FILE,
            standardize=STANDARDIZE,
            detrend=DETREND,
            low_pass=LOW_PASS,
            high_pass=HIGH_PASS,
            t_r=current_tr,
            smoothing_fwhm=SMOOTHING_FWHM,
            memory='nilearn_cache',
            memory_level=1
        )
        
        # Extract and Save
        time_series = masker.fit_transform(func_img)
        
        # Basic QC on extracted data
        if np.any(np.isnan(time_series)):
            time_series = np.nan_to_num(time_series)

        output_filename = os.path.join(OUTPUT_DIR, f"{subject_id}_timeseries.npy")
        np.save(output_filename, time_series)
        
        print(f"  ✓ Saved matrix: {time_series.shape}")
        success_count += 1
        
    except Exception as e:
        print(f"  ❌ ERROR processing {subject_id}: {str(e)}")
        failed_subjects.append((subject_id, str(e)))

print("\n" + "="*70)
print(f"PROCESS FINISHED: {success_count} subjects saved to {OUTPUT_DIR}")
print("="*70)