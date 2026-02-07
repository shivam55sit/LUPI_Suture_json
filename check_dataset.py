import os

root_dir = r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\test_data"
required_files = ["slitlamp.jpg", "anterior_224.png", "astig.json"]

valid_count = 0
invalid_count = 0
details = []

if os.path.exists(root_dir):
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if os.path.isdir(patient_path):
            missing = []
            for f in required_files:
                if not os.path.exists(os.path.join(patient_path, f)):
                    missing.append(f)
            
            if not missing:
                valid_count += 1
            else:
                invalid_count += 1
                details.append(f"{patient_id}: missing {missing}")
else:
    print(f"Root path not found: {root_dir}")

print(f"Total folders: {valid_count + invalid_count}")
print(f"Valid samples: {valid_count}")
print(f"Invalid samples: {invalid_count}")
if details:
    print("\nInvalid details (first 10):")
    for d in details[:10]:
        print(d)
