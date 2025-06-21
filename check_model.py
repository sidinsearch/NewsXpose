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

if __name__ == "__main__":
    # Get model path from command line or use default
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'ensemble_fake_news_detector.pkl'
    
    # Check current directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Check model file
    check_model_file(model_path)