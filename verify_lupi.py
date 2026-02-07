import torch
from dataset import AstigmatismLUPIDataset
from models import TeacherModel, StudentModel
from torch.utils.data import DataLoader
import os

def test_lupi_flow():
    DATASET_PATH = r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\Suture_tension_dataset"
    
    # Get a few valid patient IDs
    all_patients = sorted([
        d for d in os.listdir(DATASET_PATH) 
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])
    test_ids = all_patients[:2]
    
    print(f"Testing with patients: {test_ids}")
    
    dataset = AstigmatismLUPIDataset(DATASET_PATH, split="val", patient_ids=test_ids)
    loader = DataLoader(dataset, batch_size=2)
    
    teacher = TeacherModel()
    student = StudentModel()
    
    batch = next(iter(loader))
    limbus = batch["limbus"]
    axial = batch["axial"]
    target = batch["magnitude"]
    
    print(f"Limbus shape: {limbus.shape}")
    print(f"Axial shape: {axial.shape}")
    print(f"Target shape: {target.shape}")
    
    # Teacher forward
    t_mag, t_feat = teacher(limbus, axial)
    print(f"Teacher mag shape: {t_mag.shape}")
    print(f"Teacher feat shape: {t_feat.shape}")
    
    # Student forward
    s_mag, s_feat = student(limbus)
    print(f"Student mag shape: {s_mag.shape}")
    print(f"Student feat shape: {s_feat.shape}")
    
    assert t_feat.shape == s_feat.shape, f"Feature dimension mismatch: {t_feat.shape} vs {s_feat.shape}"
    assert t_feat.shape[1] == 4096, f"Expected 4096 features, got {t_feat.shape[1]}"
    print("✅ Verification successful! Dimensions match (4096).")

if __name__ == "__main__":
    test_lupi_flow()
