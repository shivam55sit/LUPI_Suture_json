"""Script to fix JSON files with unquoted property keys"""
import os
import re
import json
from pathlib import Path

def fix_json_content(content: str) -> str:
    """Fix unquoted property keys in JSON content."""
    # Pattern to match unquoted property keys (word followed by colon)
    # This handles cases like: k2:45.5 or target:-5.1
    pattern = r'(?<=[{,\s])(\w+)\s*:'
    
    # Replace with properly quoted keys
    fixed = re.sub(pattern, r'"\1":', content)
    
    # Try to parse and re-format as proper JSON
    try:
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        data = json.loads(fixed)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not fully parse JSON: {e}")
        return fixed

def fix_json_files_in_directory(base_dir: str):
    """Find and fix all astig.json files in subdirectories."""
    base_path = Path(base_dir)
    fixed_count = 0
    error_count = 0
    
    for json_file in base_path.rglob("astig.json"):
        try:
            # Read original content
            with open(json_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Fix the content
            fixed_content = fix_json_content(original_content)
            
            # Write back
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"[OK] Fixed: {json_file.relative_to(base_path)}")
            fixed_count += 1
            
        except Exception as e:
            print(f"[ERROR] Error fixing {json_file}: {e}")
            error_count += 1
    
    print(f"\n{'='*50}")
    print(f"Summary: Fixed {fixed_count} files, {error_count} errors")

if __name__ == "__main__":
    dataset_dir = r"c:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Suture_json\suture_tension_dataset_anterior"
    fix_json_files_in_directory(dataset_dir)
