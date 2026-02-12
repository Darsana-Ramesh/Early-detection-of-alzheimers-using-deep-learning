import os

adni_root = "../data/shiji_mri_fmri_mci_adni_2/ADNI_Clean"

unique_scans = set()

# Loop through patients
for patient in sorted(os.listdir(adni_root)):
    patient_path = os.path.join(adni_root, patient)
    if not os.path.isdir(patient_path):
        continue

    # Add scan-type subfolders
    subfolders = [d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]
    unique_scans.update(subfolders)

print("\nUnique scan types across all patients:")
for scan in sorted(unique_scans):
    print("  ", scan)

