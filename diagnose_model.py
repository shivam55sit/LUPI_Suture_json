"""
Diagnostic script to analyze model performance
"""
import torch
import numpy as np
from pathlib import Path
from dataset import AstigmatismLUPIDataset
from models import StudentModel, TeacherModel
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\Suture_tension_dataset"

def analyze_model_performance():
    """Comprehensive diagnostic analysis"""
    
    print("=" * 80)
    print("MODEL PERFORMANCE DIAGNOSTIC")
    print("=" * 80)
    
    # Load models
    print("\n1. Loading models...")
    try:
        student = StudentModel().to(DEVICE)
        teacher = TeacherModel().to(DEVICE)
        
        student.load_state_dict(torch.load("student_astigmatism_model.pth", map_location=DEVICE, weights_only=False))
        teacher.load_state_dict(torch.load("teacher_astigmatism_model.pth", map_location=DEVICE, weights_only=False))
        
        student.eval()
        teacher.eval()
        print("   [OK] Models loaded successfully")
    except Exception as e:
        print(f"   [ERROR] Error loading models: {e}")
        print("\n   -> Make sure you have trained the model first (run train.py)")
        return
    
    # Load validation data
    print("\n2. Loading validation dataset...")
    try:
        val_dataset = AstigmatismLUPIDataset(
            root_dir=DATASET_PATH,
            split="val",
            filter_outliers=False  # Don't filter for diagnosis
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print(f"   [OK] Loaded {len(val_dataset)} validation samples")
    except Exception as e:
        print(f"   [ERROR] Error loading dataset: {e}")
        return
    
    # Collect predictions
    print("\n3. Running inference on validation set...")
    predictions = {
        'teacher': [],
        'student': [],
        'ground_truth': [],
        'patient_ids': []
    }
    
    with torch.no_grad():
        for batch in val_loader:
            limbus = batch["limbus"].to(DEVICE)
            anterior = batch["anterior"].to(DEVICE)
            target = batch["magnitude"].item()
            pid = batch["patient_id"][0]
            
            # Teacher prediction
            t_pred, _ = teacher(limbus, anterior)
            t_pred = t_pred.item()
            
            # Student prediction
            s_pred, _ = student(limbus)
            s_pred = s_pred.item()
            
            predictions['teacher'].append(t_pred)
            predictions['student'].append(s_pred)
            predictions['ground_truth'].append(target)
            predictions['patient_ids'].append(pid)
    
    # Convert to numpy
    teacher_preds = np.array(predictions['teacher'])
    student_preds = np.array(predictions['student'])
    ground_truth = np.array(predictions['ground_truth'])
    
    # Calculate metrics
    print("\n4. Performance Metrics:")
    print("-" * 80)
    
    # Teacher metrics
    teacher_mae = np.mean(np.abs(teacher_preds - ground_truth))
    teacher_rmse = np.sqrt(np.mean((teacher_preds - ground_truth) ** 2))
    teacher_r2 = 1 - (np.sum((ground_truth - teacher_preds) ** 2) / 
                      np.sum((ground_truth - np.mean(ground_truth)) ** 2))
    
    print(f"\n   TEACHER MODEL (uses slitlamp + axial):")
    print(f"      MAE:  {teacher_mae:.3f} diopters")
    print(f"      RMSE: {teacher_rmse:.3f} diopters")
    print(f"      R2:   {teacher_r2:.3f}")
    
    # Student metrics
    student_mae = np.mean(np.abs(student_preds - ground_truth))
    student_rmse = np.sqrt(np.mean((student_preds - ground_truth) ** 2))
    student_r2 = 1 - (np.sum((ground_truth - student_preds) ** 2) / 
                      np.sum((ground_truth - np.mean(ground_truth)) ** 2))
    
    print(f"\n   STUDENT MODEL (uses slitlamp only):")
    print(f"      MAE:  {student_mae:.3f} diopters")
    print(f"      RMSE: {student_rmse:.3f} diopters")
    print(f"      R2:   {student_r2:.3f}")
    
    # Identify problematic predictions
    print("\n5. Worst 5 Predictions (Student Model):")
    print("-" * 80)
    errors = np.abs(student_preds - ground_truth)
    worst_indices = np.argsort(errors)[-5:][::-1]
    
    for idx in worst_indices:
        print(f"   {predictions['patient_ids'][idx]}: "
              f"GT={ground_truth[idx]:.2f}, "
              f"Pred={student_preds[idx]:.2f}, "
              f"Error={errors[idx]:.2f}")
    
    # Best predictions
    print("\n6. Best 5 Predictions (Student Model):")
    print("-" * 80)
    best_indices = np.argsort(errors)[:5]
    
    for idx in best_indices:
        print(f"   {predictions['patient_ids'][idx]}: "
              f"GT={ground_truth[idx]:.2f}, "
              f"Pred={student_preds[idx]:.2f}, "
              f"Error={errors[idx]:.2f}")
    
    # Check for common issues
    print("\n7. Diagnostic Checks:")
    print("-" * 80)
    
    # Check 1: Model predicting constant values?
    pred_std = np.std(student_preds)
    gt_std = np.std(ground_truth)
    
    print(f"\n   Prediction Variance:")
    print(f"      Ground Truth Std: {gt_std:.3f}")
    print(f"      Student Pred Std: {pred_std:.3f}")
    
    if pred_std < 0.5:
        print("      [WARNING] Student predictions have very low variance!")
        print("      -> Model may be predicting near-constant values")
        print("      -> LIKELY CAUSE: Model is underfitting")
    
    # Check 2: Predictions in reasonable range?
    gt_min, gt_max = ground_truth.min(), ground_truth.max()
    pred_min, pred_max = student_preds.min(), student_preds.max()
    
    print(f"\n   Value Ranges:")
    print(f"      Ground Truth: [{gt_min:.2f}, {gt_max:.2f}]")
    print(f"      Student Pred: [{pred_min:.2f}, {pred_max:.2f}]")
    
    if pred_max < gt_min or pred_min > gt_max:
        print("      [WARNING] Prediction range doesn't overlap with ground truth!")
        print("      -> LIKELY CAUSE: Model is not learning the correct scale")
    
    # Check 3: Distillation gap
    distill_gap = student_mae - teacher_mae
    print(f"\n   Distillation Gap:")
    print(f"      Teacher MAE: {teacher_mae:.3f}")
    print(f"      Student MAE: {student_mae:.3f}")
    print(f"      Gap: {distill_gap:.3f}")
    
    if distill_gap > 1.5:
        print(f"      [WARNING] Large gap between teacher and student!")
        print(f"      -> Student is not learning well from teacher")
    
    # Check 4: Overall performance assessment
    print(f"\n   Overall Assessment:")
    if student_mae < 1.5:
        print(f"      [EXCELLENT] MAE < 1.5 diopters")
    elif student_mae < 2.5:
        print(f"      [GOOD] MAE < 2.5 diopters")
    elif student_mae < 3.5:
        print(f"      [FAIR] MAE < 3.5 diopters (room for improvement)")
    else:
        print(f"      [POOR] MAE >= 3.5 diopters (needs improvement)")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    # Recommendations
    print("\n[RECOMMENDATIONS]")
    
    if student_mae > 3.0:
        print("\n   [CRITICAL] Student MAE is very high:")
        print("      1. Check if training completed successfully")
        print("      2. Verify data preprocessing is correct")
        print("      3. Consider training for more epochs (current: 50)")
        print("      4. Reduce dropout from 0.2 to 0.1 or 0.0")
        print("      5. Check if early stopping triggered too early (patience: 5)")
    
    if pred_std < 0.5:
        print("\n   [UNDERFITTING] Low prediction variance:")
        print("      1. Reduce dropout (0.2 -> 0.1 or 0.0)")
        print("      2. Reduce weight decay (5e-4 -> 1e-4)")
        print("      3. Increase learning rate (1e-4 -> 2e-4)")
        print("      4. Train for more epochs")
    
    if distill_gap > 1.5:
        print("\n   [POOR DISTILLATION]:")
        print("      1. Increase distillation weight (0.5 -> 0.7)")
        print("      2. Increase feature alignment weight (0.3 -> 0.5)")
        print("      3. Check if teacher model is well-trained")
    
    if student_r2 < 0.3:
        print("\n   [POOR CORRELATION]:")
        print("      1. Review data quality and labels")
        print("      2. Check for data preprocessing bugs")
        print("      3. Verify image paths are correct")
    
    # Save results
    print("\n[SAVING RESULTS]")
    with open("diagnostic_results.txt", "w") as f:
        f.write("DETAILED PREDICTION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Patient ID, Ground Truth, Teacher Pred, Student Pred, Student Error\n")
        f.write("-" * 80 + "\n")
        for i in range(len(predictions['patient_ids'])):
            f.write(f"{predictions['patient_ids'][i]}, "
                   f"{ground_truth[i]:.3f}, "
                   f"{teacher_preds[i]:.3f}, "
                   f"{student_preds[i]:.3f}, "
                   f"{errors[i]:.3f}\n")
    
    print("   [OK] Saved to: diagnostic_results.txt")

if __name__ == "__main__":
    analyze_model_performance()
