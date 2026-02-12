import os
import subprocess

MRI_KEYWORDS = ["MPRAGE"]
FMRI_KEYWORDS = ["Resting_State_fMRI", "MoCoSeries"]

DCM2NIIX = "/home/trestadmin/store/shijin_cet/dcm2niix_lnx/dcm2niix"

def is_useful_series(series_name):
    s = series_name.lower()
    if any(k.lower() in s for k in MRI_KEYWORDS):
        return "MRI"
    if any(k.lower() in s for k in FMRI_KEYWORDS):
        return "fMRI"
    return None
 
def find_dcm_folders(scan_path):
    """Return subfolders that actually contain DICOM (.dcm) files."""
    dcm_dirs = []
    for root, _, files in os.walk(scan_path):
        if any(f.endswith(".dcm") for f in files):
            dcm_dirs.append(root)
    return dcm_dirs

def get_base_filename(dcm_files):
    """Take the first DICOM filename and remove '.dcm' extension."""
    dcm_files = sorted([f for f in dcm_files if f.endswith(".dcm")])
    if not dcm_files:
        return None
    return os.path.splitext(dcm_files[0])[0]  # remove .dcm

def convert_dcm_to_nii(input_dir, output_dir):
    for root, dirs, _ in os.walk(input_dir):
        for d in dirs:
            modality = is_useful_series(d)
            if modality is None:
                continue

            scan_path = os.path.join(root, d)
            dcm_folders = find_dcm_folders(scan_path)
            if not dcm_folders:
                continue

            # patient ID
            parts = os.path.normpath(scan_path).split(os.sep)
            patient_id = parts[parts.index("ADNI") + 1]

            for dcm_dir in dcm_folders:
                dcm_files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
                if not dcm_files:
                    continue

                prefix_name = get_base_filename(dcm_files)
                if not prefix_name:
                    continue

                output_subdir = os.path.join(output_dir, patient_id, d)
                os.makedirs(output_subdir, exist_ok=True)

                command = [
                    DCM2NIIX,
                    "-z", "n",       # no compression -> .nii
                    "-f", prefix_name,
                    "-o", output_subdir,
                    dcm_dir
                ]
                try:
                    print(f"[INFO] Converting {dcm_dir} ({modality}) -> {output_subdir}/{prefix_name}.nii")
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] Could not convert {dcm_dir}: {e}")

# Example usage
input_directory = "../data/shij_mri_fmri_lmci_adni_2/ADNI"
output_directory = "../data/shij_mri_fmri_lmci_adni_2/ADNI_Clean"

convert_dcm_to_nii(input_directory, output_directory)

