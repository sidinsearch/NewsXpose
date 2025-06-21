import os
import re

def fix_app_py():
    """
    Update app.py to suppress the warning message about incompatible dtype.
    """
    app_py_path = 'app.py'
    
    if not os.path.exists(app_py_path):
        print(f"Error: {app_py_path} not found")
        return False
    
    with open(app_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if we need to add the warning suppression
    if "warnings.filterwarnings" in content and "node array from the pickle has an incompatible dtype" in content:
        print("Warning suppression already added to app.py")
        return False
    
    # Add import warnings if not already present
    if "import warnings" not in content:
        content = re.sub(
            r'(import .*?\n)',
            r'\1import warnings\n',
            content,
            count=1
        )
    
    # Add warning suppression after the imports but before the first function or class
    warning_code = """
# Suppress warnings about incompatible dtype in node arrays
warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")
"""
    
    # Find a good place to insert the warning suppression
    # After imports but before the first function or class definition
    import_section_end = 0
    for match in re.finditer(r'import .*?\n|from .*? import .*?\n', content):
        if match.end() > import_section_end:
            import_section_end = match.end()
    
    if import_section_end > 0:
        # Insert after the last import statement
        new_content = content[:import_section_end] + warning_code + content[import_section_end:]
        
        # Write the updated content back to the file
        with open(app_py_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Successfully updated {app_py_path} to suppress warnings")
        return True
    else:
        print(f"Could not find a suitable place to insert warning suppression in {app_py_path}")
        return False

if __name__ == "__main__":
    fix_app_py()