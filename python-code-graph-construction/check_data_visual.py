import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- CONFIGURATION ---
# Path to your output folder
DATA_DIR = r"C:\Users\DELL\Desktop\graph-construction\Timeseries_Output"

# Pick ONE file to check (copy the name from your folder)
FILE_NAME = "swausub-002S1155_timeseries.npy" 
# ---------------------

file_path = os.path.join(DATA_DIR, FILE_NAME)

if not os.path.exists(file_path):
    # If specific file not found, grab the first one automatically
    files = list(Path(DATA_DIR).glob("*.npy"))
    if files:
        file_path = str(files[0])
        print(f"File not found, checking first available file: {files[0].name}")

# 1. Load the data
data = np.load(file_path)

print(f"Checking File: {file_path}")
print(f"Data Shape: {data.shape} (Timepoints x Regions)")
print(f"Min Value: {data.min():.4f}")
print(f"Max Value: {data.max():.4f}")

# 2. Plot Heatmap
plt.figure(figsize=(12, 6))
# We transpose (.T) so Time is on the X-axis and Regions are on Y-axis
plt.imshow(data.T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Signal Intensity (Z-score)')
plt.title(f"ROI Time Series: {Path(file_path).name}")
plt.xlabel("Time (TRs)")
plt.ylabel("Brain Regions (ROIs)")
plt.tight_layout()

# Show the plot
print("Opening visualization...")
plt.show()