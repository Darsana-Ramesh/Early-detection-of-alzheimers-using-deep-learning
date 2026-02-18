import sys
import torch
import joblib
import nilearn
import os

def verify():
    print("--- NeuroVista Environment Check ---")
    
    # 1. Check Libraries
    libs = ['flask', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'nilearn', 'torch', 'torch_geometric', 'statsmodels']
    missing = []
    for lib in libs:
        try:
            __import__(lib)
            print(f"✅ {lib} is installed.")
        except ImportError:
            print(f"❌ {lib} is MISSING.")
            missing.append(lib)

    # 2. Check Files
    files = ['neurovista_model.pth', 'scaler.pkl', 'ROI_MNI_V4.nii']
    for f in files:
        if os.path.exists(f):
            print(f"✅ File found: {f}")
        else:
            print(f"❌ File NOT found: {f}")

    # 3. Test Model Loading (CPU)
    try:
        from app import NeuroVistaPrecisionNet
        model = NeuroVistaPrecisionNet(node_in=32, vector_in=1024)
        model.load_state_dict(torch.load("neurovista_model.pth", map_location='cpu'))
        print("✅ GNN Model loaded into memory successfully.")
    except Exception as e:
        print(f"⚠️ Model loading test failed: {e}")

    if not missing:
        print("\n🚀 ALL SYSTEMS GO! You can run 'python app.py' now.")
    else:
        print(f"\n🛑 Please install missing libs: pip install {' '.join(missing)}")

if __name__ == "__main__":
    verify()