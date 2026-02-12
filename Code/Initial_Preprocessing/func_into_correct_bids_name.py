import os
from pathlib import Path

# --- ! IMPORTANT ! ---
# Change this to your dataset folder path
BIDS_DIR = r"C:\Users\DELL\Desktop\4th-year-project\ADNI_BIDS_MCI_new"
# Example for others:
# BIDS_DIR = r"D:\datasets\ADNI_BIDS_MCI"
# --- ! --------- ! ---

# Keywords to identify functional scans
FMRI_KEYWORDS = ["rsfmri", "fcmri", "rest", "mocoseries"]

print(f"--- Starting BIDS rename process for: {BIDS_DIR} ---")

# Find all func directories for subjects
func_dirs = list(Path(BIDS_DIR).glob("sub-*/func"))

if not func_dirs:
    print(f"Error: No 'sub-*/func' directories found in {BIDS_DIR}")
    print("Please check the BIDS_DIR path and folder structure.")
    exit()

for func_dir in func_dirs:
    print(f"\nProcessing: {func_dir}")
    
    # Collect files that need renaming
    files_to_rename = []
    
    for nii_file in func_dir.glob("*.nii"):
        filename = nii_file.name.lower()
        is_func = any(keyword in filename for keyword in FMRI_KEYWORDS)
        is_bids = "_task-" in filename and "_bold" in filename
        
        if is_func and not is_bids:
            files_to_rename.append(nii_file)
    
    if not files_to_rename:
        print("... No non-compliant fMRI files found. Skipping.")
        continue

    files_to_rename.sort()
    
    # Rename each file (nii + json)
    for run_index, old_nii_file in enumerate(files_to_rename, 1):
        subject_id = old_nii_file.name.split("_")[0]
        
        new_nii_name = f"{subject_id}_task-rest_run-{run_index:02d}_bold.nii"
        new_json_name = f"{subject_id}_task-rest_run-{run_index:02d}_bold.json"
        
        old_json_file = old_nii_file.with_suffix(".json")
        
        new_nii_path = func_dir / new_nii_name
        new_json_path = func_dir / new_json_name
        
        try:
            # Rename .nii
            old_nii_file.rename(new_nii_path)
            print(f"  ✅ {old_nii_file.name} → {new_nii_path.name}")
            
            # Rename .json (if present)
            if old_json_file.exists():
                old_json_file.rename(new_json_path)
                print(f"  ✅ {old_json_file.name} → {new_json_path.name}")
            else:
                print(f"  ⚠️  No matching .json for {old_nii_file.name}")
                
        except Exception as e:
            print(f"  ❌ Error renaming {old_nii_file.name}: {e}")

print("\n--- ✅ All renaming completed successfully! ---")
print("You can now load this dataset into CONN or any BIDS-compatible tool.")
