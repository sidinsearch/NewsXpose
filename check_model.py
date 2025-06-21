import os
import pickle
import joblib
import warnings
import sys

def check_model_file(model_path):
    """Check if a model file exists and can be loaded."""
    print(f"Checking model file: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"Model file size: {file_size} bytes")
    if file_size == 0:
        print(f"Error: Model file is empty")
        return False
    
    # Try to load with pickle
    print("Attempting to load with pickle...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Successfully loaded model with pickle")
            
            # Check if it's a tuple with model and vectorizer
            if isinstance(model, tuple) and len(model) == 2:
                print(f"Model is a tuple with {len(model)} elements")
                print(f"Element 1 type: {type(model[0]).__name__}")
                print(f"Element 2 type: {type(model[1]).__name__}")
            else:
                print(f"Model type: {type(model).__name__}")
            
            return True
    except Exception as e:
        print(f"Error loading with pickle: {str(e)}")
    
    # Try to load with joblib
    print("Attempting to load with joblib...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")
            model = joblib.load(model_path)
            print(f"Successfully loaded model with joblib")
            
            # Check if it's a tuple with model and vectorizer
            if isinstance(model, tuple) and len(model) == 2:
                print(f"Model is a tuple with {len(model)} elements")
                print(f"Element 1 type: {type(model[0]).__name__}")
                print(f"Element 2 type: {type(model[1]).__name__}")
            else:
                print(f"Model type: {type(model).__name__}")
            
            return True
    except Exception as e:
        print(f"Error loading with joblib: {str(e)}")
    
    print(f"Failed to load model with both pickle and joblib")
    return False

def find_model_file(base_name):
    """Find a model file in multiple possible locations."""
    # Define possible model paths
    possible_paths = [
        f'/app/models/{base_name}',  # Docker container path
        f'models/{base_name}',       # Subdirectory path
        base_name                    # Root directory path
    ]
    
    # Find the first existing model path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    print(f"Model file not found: {base_name}")
    return None

if __name__ == "__main__":
    # Get model base name from command line or use default
    model_base_name = sys.argv[1] if len(sys.argv) > 1 else 'ensemble_fake_news_detector.pkl'
    
    # Check current directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Check if models directory exists
    if os.path.exists('/app/models'):
        print(f"Files in /app/models: {os.listdir('/app/models')}")
    elif os.path.exists('models'):
        print(f"Files in models: {os.listdir('models')}")
    
    # Find model file
    model_path = find_model_file(model_base_name)
    
    if model_path:
        # Check model file
        check_model_file(model_path)
    else:
        print(f"Model file not found in any of the expected locations")