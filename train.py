import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import AstigmatismLUPIDataset
from models import TeacherModel, StudentModel
import os
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50  # Increased for better convergence
BATCH_SIZE = 4  # Smaller batch for better generalization
LEARNING_RATE = 1e-4  # Increased from 1e-4
DISTILL_WEIGHT = 0.5  # Increased to leverage teacher more
FEAT_WEIGHT = 0.6     # Increased for better feature alignment
NUM_WORKERS = 0  # Set to 0 for Windows stability (data is cached in RAM anyway)

# Dataset path
DATASET_PATH = r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\Suture_tension_dataset"


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train():
    print(f"Using device: {DEVICE}")
    
    # ---- Load dataset ----
    all_patients = sorted([
        d for d in os.listdir(DATASET_PATH) 
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])
    
    import random
    random.seed(42)
    random.shuffle(all_patients)
    
    split_idx = int(0.9 * len(all_patients))
    train_ids = all_patients[:split_idx]
    val_ids = all_patients[split_idx:]
    
    print(f"Total folders: {len(all_patients)}")
    
    # Create datasets
    train_dataset = AstigmatismLUPIDataset(DATASET_PATH, split="train", patient_ids=train_ids)
    val_dataset = AstigmatismLUPIDataset(DATASET_PATH, split="val", patient_ids=val_ids)
    # DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if NUM_WORKERS > 0 else False,  # Keep workers alive
        prefetch_factor=2 if NUM_WORKERS > 0 else None  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    # ---- Initialize models ----
    teacher = TeacherModel().to(DEVICE)
    student = StudentModel().to(DEVICE)

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        list(teacher.parameters()) + list(student.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4  # Reduced from 5e-4 to reduce over-regularization
    )
    
    # Advanced scheduler: Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # Losses
    # Huber loss is more robust to outliers and clinical noise
    criterion_reg = torch.nn.HuberLoss(delta=1.0) 
    criterion_feat = torch.nn.MSELoss()

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=10, verbose=True)  # Increased patience

    for epoch in range(EPOCHS):
        # ---- Training ----
        teacher.train()
        student.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch_idx, batch in enumerate(pbar):
            try:
                limbus = batch["limbus"].to(DEVICE, non_blocking=True)
                anterior = batch["anterior"].to(DEVICE, non_blocking=True)
                target_mag = batch["magnitude"].to(DEVICE, non_blocking=True)

                # ---- Forward ----
                teacher_mag, teacher_feat = teacher(limbus, anterior)
                student_mag, student_feat = student(limbus)

                # ---- Losses ----
                teacher_loss = criterion_reg(teacher_mag, target_mag)
                distill_loss = criterion_reg(student_mag, teacher_mag.detach())
                feat_loss = criterion_feat(student_feat, teacher_feat.detach())
                loss = teacher_loss + DISTILL_WEIGHT * distill_loss + FEAT_WEIGHT * feat_loss

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}: {e}")
                if "out of memory" in str(e):
                    print("WARNING: Out of memory. Clearing cache and skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        teacher.eval()
        student.eval()
        val_loss = 0
        val_teacher_mag_mae = 0
        val_student_mag_mae = 0
        
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for batch in vbar:
                limbus = batch["limbus"].to(DEVICE)
                anterior = batch["anterior"].to(DEVICE)
                target_mag = batch["magnitude"].to(DEVICE)

                t_mag, _ = teacher(limbus, anterior)
                s_mag, _ = student(limbus)

                val_loss += criterion_reg(t_mag, target_mag).item()
                val_teacher_mag_mae += F.l1_loss(t_mag, target_mag).item()
                val_student_mag_mae += F.l1_loss(s_mag, target_mag).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_t_mag_mae = val_teacher_mag_mae / len(val_loader)
        avg_s_mag_mae = val_student_mag_mae / len(val_loader)
        
        # Update scheduler
        scheduler.step()

        # Use tqdm.write to keep progress bars clean
        tqdm.write(f"\nSummary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                   f"T MAE: {avg_t_mag_mae:.2f} | S MAE: {avg_s_mag_mae:.2f}")

        # ---- Save best models ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(teacher.state_dict(), "teacher_astigmatism_model.pth")
            torch.save(student.state_dict(), "student_astigmatism_model.pth")
            tqdm.write(f"  ✅ Saved best models (val_loss: {best_val_loss:.4f})")
            
        # Check early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered. Training halted.")
            break
            
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("Optimization Complete!")


if __name__ == "__main__":
    train()
