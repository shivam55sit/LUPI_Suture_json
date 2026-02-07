import os
import json
import re
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class AstigmatismLUPIDataset(Dataset):
    def __init__(self, root_dir, split="train", patient_ids=None, filter_outliers=True):
        """
        LUPI Dataset for Astigmatism Prediction (Optimized)
        
        Args:
            root_dir: Path to dataset directory
            split: 'train' or 'val'
            patient_ids: Optional list of patient IDs to use
            filter_outliers: If True, removes outliers using IQR method (1.5×IQR)
        """
        self.root_dir = root_dir
        self.split = split

        if patient_ids is not None:
            active_ids = patient_ids
        else:
            active_ids = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])

        # Minimal photometric transforms (applied only to slitlamp)
        # No geometric transforms to preserve spatial correspondence with axial map
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1) if split == "train" else None

        # Pre-load/Validate samples and cache metadata
        self.samples = []
        self.cache = {} # In-memory cache for images
        
        print(f"Loading {split} dataset metadata...")
        for pid in active_ids:
            patient_dir = os.path.join(root_dir, pid)
            limbus_path = os.path.join(patient_dir, "slitlamp_limbus.png")
            anterior_path = os.path.join(patient_dir, "anterior_224.png")
            json_path = os.path.join(patient_dir, "astig.json")

            if all(os.path.exists(p) for p in [limbus_path, anterior_path, json_path]):
                try:
                    meta = self._parse_json(json_path)
                    magnitude = abs(float(meta.get("target", 0)))
                    
                    self.samples.append({
                        "pid": pid,
                        "limbus_path": limbus_path,
                        "anterior_path": anterior_path,
                        "magnitude": magnitude
                    })

                    # Optional: Pre-load images into cache during initialization
                    # Since dataset is small (~85-95 samples), we can afford this RAM usage
                    self.cache[pid] = {
                        "limbus": self._load_image(limbus_path),
                        "anterior": self._load_image(anterior_path)
                    }
                except Exception as e:
                    print(f"Warning: Skipping {pid} due to JSON error: {e}")
            else:
                pass
        
        print(f"  Loaded {len(self.samples)} valid samples (and cached in RAM).")
        
        # Filter outliers using IQR method (only for training)
        if filter_outliers and split == "train" and len(self.samples) > 0:
            magnitudes = np.array([s["magnitude"] for s in self.samples])
            q1 = np.percentile(magnitudes, 25)
            q3 = np.percentile(magnitudes, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter samples
            filtered_samples = []
            filtered_pids = []
            for sample in self.samples:
                if lower_bound <= sample["magnitude"] <= upper_bound:
                    filtered_samples.append(sample)
                else:
                    filtered_pids.append(sample["pid"])
                    # Remove from cache
                    if sample["pid"] in self.cache:
                        del self.cache[sample["pid"]]
            
            if len(filtered_pids) > 0:
                print(f"  ⚠️  Filtered {len(filtered_pids)} outlier(s): {filtered_pids}")
                print(f"  ✅ Remaining samples: {len(filtered_samples)}")
            
            self.samples = filtered_samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path, size=(224, 224)):
        """Load and preprocess an image to Tensor (with ImageNet normalization)"""
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = torch.from_numpy(img).permute(2, 0, 1) # (C, H, W)
        return img

    def _parse_json(self, json_path):
        """Parse JSON file with flexible format"""
        with open(json_path, "r") as f:
            content = f.read()
        
        # Handle non-standard JSON (missing quotes around keys)
        content = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', content)
        
        try:
            meta = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON {json_path}: {e}")
        
        return meta

    def apply_transforms(self, limbus, anterior=None):
        """Apply minimal transforms to preserve spatial correspondence"""
        if self.split == "train":
            # Only apply photometric transforms to limbus (not anterior)
            # This preserves spatial alignment between limbus and anterior map
            if self.color_jitter and random.random() > 0.5:
                limbus = self.color_jitter(limbus)
        
        return limbus, anterior

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["pid"]

        # ---- Load images (from cache) ----
        if pid in self.cache:
            limbus = self.cache[pid]["limbus"].clone()
            anterior = self.cache[pid]["anterior"].clone()
        else:
            # Fallback if not cached (should not happen with current init)
            limbus = self._load_image(sample["limbus_path"])
            anterior = self._load_image(sample["anterior_path"])
        
        # ---- Apply transforms ----
        limbus, anterior = self.apply_transforms(limbus, anterior)

        return {
            "limbus": limbus,
            "anterior": anterior,
            "magnitude": torch.tensor(sample["magnitude"], dtype=torch.float32),
            "patient_id": pid
        }
