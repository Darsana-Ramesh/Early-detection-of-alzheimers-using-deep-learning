import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIGURATION =================
# 1. INPUT: Where your .npy graph files are
INPUT_DIR = r"C:\NEURO_Trial\Graph_Output"

# 2. OUTPUT: Where to save the images
OUTPUT_DIR = r"C:\NEURO_Trial\Graph_Plots"
# =================================================

print("--- Generatng Pearson Correlation Heatmaps ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all correlation matrix files
matrix_files = list(Path(INPUT_DIR).glob("*_corr_matrix.npy"))

if not matrix_files:
    print(f"❌ No .npy files found in {INPUT_DIR}")
    exit()

print(f"Found {len(matrix_files)} graphs. Generating images...")

for i, mat_file in enumerate(matrix_files):
    # Load the matrix
    correlation_matrix = np.load(mat_file)
    subject_id = mat_file.name.split('_corr_matrix')[0]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # vmin=-1, vmax=1 ensures the colors map correctly from -1 (Blue) to +1 (Red)
    plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    plt.colorbar(label='Pearson Correlation (r)')
    plt.title(f"Functional Connectivity: {subject_id}")
    plt.xlabel("Brain Regions (ROIs 1-116)")
    plt.ylabel("Brain Regions (ROIs 1-116)")
    
    # Save as PNG
    save_path = os.path.join(OUTPUT_DIR, f"{subject_id}_heatmap.png")
    plt.savefig(save_path)
    plt.close() # Close figure to free memory
    
    print(f"[{i+1}/{len(matrix_files)}] Saved: {save_path}")

print("\n" + "="*50)
print("DONE! You can view your graphs here:")
print(f"{OUTPUT_DIR}")
print("="*50)