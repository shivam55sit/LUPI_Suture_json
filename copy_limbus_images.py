import os
import shutil
from pathlib import Path

def migrate_limbus_images():
    source_root = Path(r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\03Data")
    target_root = Path(r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\test_data")
    
    if not source_root.exists():
        print(f"Error: Source directory {source_root} does not exist.")
        return
    
    if not target_root.exists():
        print(f"Error: Target directory {target_root} does not exist.")
        return
    
    copied_count = 0
    skipped_count = 0
    
    # Iterate through each patient folder in 03Data
    for patient_folder in source_root.iterdir():
        if patient_folder.is_dir():
            pid = patient_folder.name
            source_img = patient_folder / "limbus.jpg"
            target_patient_dir = target_root / pid
            
            if source_img.exists() and target_patient_dir.is_dir():
                target_img = target_patient_dir / "limbus.jpg"
                shutil.copy2(source_img, target_img)
                copied_count += 1
                # print(f"Copied: {pid}/limbus.jpg")
            else:
                skipped_count += 1
                # if not target_patient_dir.is_dir():
                #     print(f"Skipped (No target folder): {pid}")
    
    print("-" * 30)
    print(f"Migration finished.")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped: {skipped_count}")
    print("-" * 30)

if __name__ == "__main__":
    migrate_limbus_images()
