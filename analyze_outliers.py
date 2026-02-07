import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\Suture_tension_dataset"

def parse_json(json_path):
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

# Collect all target values
targets = []
patient_ids = []

all_patients = sorted([
    d for d in os.listdir(DATASET_PATH) 
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

for pid in all_patients:
    patient_dir = os.path.join(DATASET_PATH, pid)
    json_path = os.path.join(patient_dir, "astig.json")
    
    if os.path.exists(json_path):
        try:
            meta = parse_json(json_path)
            magnitude = abs(float(meta.get("target", 0)))
            targets.append(magnitude)
            patient_ids.append(pid)
        except Exception as e:
            print(f"Warning: Skipping {pid} due to error: {e}")

targets = np.array(targets)

# Calculate statistics
mean = np.mean(targets)
std = np.std(targets)
median = np.median(targets)
q1 = np.percentile(targets, 25)
q3 = np.percentile(targets, 75)
iqr = q3 - q1

# IQR method for outliers
lower_bound_iqr = q1 - 1.5 * iqr
upper_bound_iqr = q3 + 1.5 * iqr

# Z-score method for outliers
lower_bound_zscore = mean - 3 * std
upper_bound_zscore = mean + 3 * std

print("=" * 60)
print("ASTIGMATISM TARGET DISTRIBUTION ANALYSIS")
print("=" * 60)
print(f"\nTotal samples: {len(targets)}")
print(f"\nBasic Statistics:")
print(f"  Mean: {mean:.2f}")
print(f"  Std Dev: {std:.2f}")
print(f"  Median: {median:.2f}")
print(f"  Min: {np.min(targets):.2f}")
print(f"  Max: {np.max(targets):.2f}")

print(f"\nQuartiles:")
print(f"  Q1 (25%): {q1:.2f}")
print(f"  Q2 (50%): {median:.2f}")
print(f"  Q3 (75%): {q3:.2f}")
print(f"  IQR: {iqr:.2f}")

print(f"\nOutlier Detection (IQR Method - 1.5×IQR):")
print(f"  Lower bound: {lower_bound_iqr:.2f}")
print(f"  Upper bound: {upper_bound_iqr:.2f}")
outliers_iqr = (targets < lower_bound_iqr) | (targets > upper_bound_iqr)
print(f"  Outliers: {np.sum(outliers_iqr)} samples")

print(f"\nOutlier Detection (Z-Score Method - 3σ):")
print(f"  Lower bound: {lower_bound_zscore:.2f}")
print(f"  Upper bound: {upper_bound_zscore:.2f}")
outliers_zscore = (targets < lower_bound_zscore) | (targets > upper_bound_zscore)
print(f"  Outliers: {np.sum(outliers_zscore)} samples")

# List outliers
if np.sum(outliers_iqr) > 0:
    print(f"\nOutlier Patients (IQR Method):")
    for i, is_outlier in enumerate(outliers_iqr):
        if is_outlier:
            print(f"  {patient_ids[i]}: {targets[i]:.2f}")

# Create histogram
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(targets, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
plt.axvline(upper_bound_iqr, color='orange', linestyle='--', label=f'IQR Upper: {upper_bound_iqr:.2f}')
plt.xlabel('Astigmatism Magnitude')
plt.ylabel('Frequency')
plt.title('Target Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(targets, vert=True)
plt.ylabel('Astigmatism Magnitude')
plt.title('Box Plot (Outliers shown as circles)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization to: target_distribution_analysis.png")
print("=" * 60)
