import os
import re
import glob

def find_model_loading_code(directory='.'):
    """Find files that likely load scikit-learn models"""
    python_files = glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)
    model_loading_files = []
    
    patterns = [
        r'pickle\.load',
        r'joblib\.load',
        r'load_model',
        r'from\s+sklearn',
        r'import\s+pickle',
        r'import\s+joblib'
    ]
    
    for file_path in python_files:
        # Skip this script itself
        if os.path.basename(file_path) == 'update_model_loading.py' or os.path.basename(file_path) == 'model_utils.py':
            continue
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read()
                for pattern in patterns:
                    if re.search(pattern, content):
                        model_loading_files.append(file_path)
                        break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return model_loading_files

def update_file(file_path):
    """Update a file to use the safe_load_model function"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check if the file already imports from model_utils
    if 'from model_utils import' in content:
        print(f"File {file_path} already updated")
        return False
    
    # Add import for model_utils
    import_pattern = r'(import\s+[^\n]+\n|from\s+[^\n]+\n)'
    imports = re.findall(import_pattern, content)
    
    if imports:
        last_import = imports[-1]
        last_import_pos = content.rfind(last_import) + len(last_import)
        new_content = (
            content[:last_import_pos] + 
            "\nfrom model_utils import safe_load_model, is_model_compatible\n" + 
            content[last_import_pos:]
        )
    else:
        new_content = "from model_utils import safe_load_model, is_model_compatible\n\n" + content
    
    # Replace pickle.load calls with safe_load_model
    pickle_pattern = r'pickle\.load\(\s*open\(\s*([^,]+),\s*[\'"]rb[\'"]\s*\)\s*\)'
    new_content = re.sub(pickle_pattern, r'safe_load_model(\1)', new_content)
    
    # Replace joblib.load calls
    joblib_pattern = r'joblib\.load\(\s*([^)]+)\s*\)'
    new_content = re.sub(joblib_pattern, r'safe_load_model(\1)', new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
        return True
    else:
        print(f"No changes needed for {file_path}")
        return False

def main():
    model_files = find_model_loading_code()
    
    if not model_files:
        print("No files found that load models.")
        return
    
    print(f"Found {len(model_files)} files that might load models:")
    for file in model_files:
        print(f"  - {file}")
    
    print("\nUpdating files to use safe_load_model...")
    updated = 0
    for file in model_files:
        if update_file(file):
            updated += 1
    
    print(f"\nUpdated {updated} files.")
    print("\nNext steps:")
    print("1. Review the changes to ensure they're correct")
    print("2. Test your application locally")
    print("3. Rebuild your Docker container")
    print("4. Deploy to Render")

if __name__ == "__main__":
    main()