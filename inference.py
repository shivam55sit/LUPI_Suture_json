import os
import cv2
import torch
import numpy as np
import pandas as pd
import json
import re
from models import StudentModel
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_limbus(image_path, size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    
    # ImageNet Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

def load_model():
    model = StudentModel()
    model_path = r"C:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\student_astigmatism_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weight not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def parse_astig_json(json_path):
    """Parse JSON file with flexible format (handles missing quotes around keys)"""
    if not os.path.exists(json_path):
        return None, None
    
    with open(json_path, "r") as f:
        content = f.read()
    
    # Handle non-standard JSON (missing quotes around keys)
    content = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', content)
    
    try:
        meta = json.loads(content)
        magnitude = abs(float(meta.get("target", 0)))
        axis = float(meta.get("axis", 0))
        return magnitude, axis
    except Exception as e:
        print(f"Warning: Failed to parse JSON {json_path}: {e}")
        return None, None

def batch_inference(parent_folder):
    model = load_model()
    results = []
    parent_path = Path(parent_folder)

    # Get all patient folders
    patient_folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    if not patient_folders:
        print(f"No subfolders found in {parent_folder}")
        return

    print(f"Found {len(patient_folders)} patient folders. Starting inference...")

    for pid in patient_folders:
        patient_dir = parent_path / pid
        limbus_path = patient_dir / "slitlamp_limbus.png"
        json_path = patient_dir / "astig.json"

        if not limbus_path.exists():
            print(f"Skipping {pid}: slitlamp_limbus.png not found.")
            continue

        # 1. Prediction
        try:
            x = preprocess_limbus(str(limbus_path))
            with torch.no_grad():
                pred_mag_tensor, _ = model(x) # Extract magnitude, ignore features
                pred_mag = pred_mag_tensor.cpu().item()
        except Exception as e:
            print(f"Error predicting for {pid}: {e}")
            continue

        # 2. Get Actual Values
        actual_mag, _ = parse_astig_json(json_path)

        # 3. Store Results
        results.append({
            "Patient_ID": pid,
            "Predicted_Mag": round(pred_mag, 4),
            "Actual_Mag": actual_mag if actual_mag is not None else "N/A"
        })

    # Create Excel
    if results:
        df = pd.DataFrame(results)
        output_path =  "inference_results.xlsx"
        df.to_excel(output_path, index=False)
        print(f"Successfully saved results to {output_path}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    # USER: Provide the path to the parent folder containing patient subfolders
    parent_data_path = r"C:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\test_data"
    
    if os.path.exists(parent_data_path):
        batch_inference(parent_data_path)
    else:
        print(f"Parent folder path does not exist: {parent_data_path}")

