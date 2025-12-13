import os

# 1. SET THE PATH TO YOUR MAIN FOLDER
# Use 'r' before the string to handle backslashes correctly in Windows.
main_path = r"C:\Users\DELL\Desktop\4th-year-project\ADNI_BIDS_MCI_new"

# --- DO NOT EDIT BELOW THIS LINE ---

# Lists to hold the names of folders with issues
missing_both = []
missing_anat_only = []
missing_func_only = []

print(f"🔍 Checking folders inside: {main_path}\n")

# Check if the main directory exists
if not os.path.isdir(main_path):
    print(f"Error: The directory '{main_path}' does not exist. Please check the path.")
else:
    # Get all items in the main directory that are folders and start with 'sub-'
    subject_folders = [
        f for f in os.listdir(main_path) 
        if os.path.isdir(os.path.join(main_path, f)) and f.startswith('sub-')
    ]

    # Loop through each subject folder
    for folder_name in subject_folders:
        current_subject_path = os.path.join(main_path, folder_name)
        
        # Check if 'anat' and 'func' subfolders exist
        anat_exists = os.path.isdir(os.path.join(current_subject_path, "anat"))
        func_exists = os.path.isdir(os.path.join(current_subject_path, "func"))
        
        # Categorize the folder based on what's missing
        if not anat_exists and not func_exists:
            missing_both.append(folder_name)
        elif not anat_exists:
            missing_anat_only.append(folder_name)
        elif not func_exists:
            missing_func_only.append(folder_name)

    # --- Print the results ---
    print("--- ✅ Scan Complete! Here is the report: ---\n")

    if not missing_both and not missing_anat_only and not missing_func_only:
        print("🎉 Excellent! All subject folders have both 'anat' and 'func' subfolders.")
    else:
        if missing_func_only:
            print("Folders MISSING 'func':")
            for folder in missing_func_only:
                print(f"  - {folder}")
            print("-" * 20)
            
        if missing_anat_only:
            print("Folders MISSING 'anat':")
            for folder in missing_anat_only:
                print(f"  - {folder}")
            print("-" * 20)

        if missing_both:
            print("Folders MISSING BOTH 'anat' AND 'func':")
            for folder in missing_both:
                print(f"  - {folder}")
            print("-" * 20)
