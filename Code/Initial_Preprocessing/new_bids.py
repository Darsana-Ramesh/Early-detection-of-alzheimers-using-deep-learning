import os
import shutil
import json
from pathlib import Path
import csv

RAW_CLEAN_DIR = "../data/shiji_mri_fmri_mci_adni_2/ADNI_Clean"
BIDS_DIR = "../data/ADNI_BIDS"
os.makedirs(BIDS_DIR, exist_ok=True)

# ---- dataset_description.json ----
dataset_description = {
    "Name": "ADNI Converted Dataset",
    "BIDSVersion": "1.9.0",
    "DatasetType": "raw",
    "Authors": ["Your Name"]
}
with open(Path(BIDS_DIR) / "dataset_description.json", "w") as f:
    json.dump(dataset_description, f, indent=4)

def write_json(json_path, content):
    """Write sidecar JSON if it doesn’t exist."""
    if not json_path.exists():
        with open(json_path, "w") as f:
            json.dump(content, f, indent=4)

def move_to_bids(subj_id, modality, nii_file):
    """Map modality to BIDS folder and create JSON sidecar."""
    bids_subj = f"sub-{subj_id.replace('_', '')}"
    subj_dir = Path(BIDS_DIR) / bids_subj
    print("inside move_to_bids")

    if modality in ["MPRAGE", "MPRAGE_GRAPPA2", "MPRAGE_SENSE2"]:
        out_dir = subj_dir / "anat"
        out_name = f"{bids_subj}_T1w.nii"
        json_name = f"{bids_subj}_T1w.json"
        json_content = {
            "Modality": "MR",
            "SeriesDescription": "T1w MPRAGE",
            "TaskName": "n/a"
        }
        print("inside if")
    elif modality in ["MoCoSeries", "Resting_State_fMRI"]:
        out_dir = subj_dir / "func"
        out_name = f"{bids_subj}_task-rest_bold.nii"
        json_name = f"{bids_subj}_task-rest_bold.json"
        json_content = {
            "TaskName": "rest"
        }
        print("elif")
    else:
        out_dir = subj_dir / "misc"
        out_name = f"{bids_subj}_{modality}.nii"
        json_name = f"{bids_subj}_{modality}.json"
        json_content = {"Modality": "MR", "SeriesDescription": modality}
        print("else")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(nii_file, out_dir / out_name)
    write_json(out_dir / json_name, json_content)

    print(f"[BIDS] {nii_file} -> {out_dir/out_name}")

def write_participants(bids_dir, subj_ids):
    tsv_file = Path(bids_dir) / "participants.tsv"
    with open(tsv_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["participant_id"])
        for sid in subj_ids:
            writer.writerow([f"sub-{sid.replace('_','')}"])  # ADNI style → BIDS ID
    print(f"[INFO] Wrote participants.tsv at {tsv_file}")


# ---- Process each subject ----
for subj in Path(RAW_CLEAN_DIR).iterdir():
    if not subj.is_dir():
        continue

    subj_id = subj.name  # e.g. 067_S_4212

    # search recursively inside subject folders (MPRAGE, MoCoSeries, etc.)
    for nii_file in subj.rglob("*.nii*"):
        if not nii_file.is_file():
            continue

        fname = nii_file.name
        parent_dir = nii_file.parent.name  # modality folder name, e.g. "MPRAGE"

        # Map modality by folder name (not filename!)
        if "MPRAGE" in parent_dir:
            modality = "MPRAGE"
        elif "MoCoSeries" in parent_dir:
            modality = "MoCoSeries"
        elif "Resting_State_fMRI" in parent_dir:
            modality = "Resting_State_fMRI"
        else:
            modality = "Other"

        move_to_bids(subj_id, modality, nii_file)

all_subjects = [subj.name for subj in Path(RAW_CLEAN_DIR).iterdir() if subj.is_dir()]
write_participants(BIDS_DIR, all_subjects)



