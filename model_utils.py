import os
import pickle
import warnings
import numpy as np
from sklearn.base import BaseEstimator

def find_model_file(base_name):
    """
    Find a model file in multiple possible locations.
    
    Args:
        base_name (str): Base name of the model file (e.g., 'ensemble_fake_news_detector.pkl')
        
    Returns:
        str: Full path to the model file if found, None otherwise
    """
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

def safe_load_model(model_path):
    """
    Safely load a scikit-learn model with compatibility handling.
    
    This function suppresses warnings about incompatible dtype in node arrays
    that can occur when loading models trained with different scikit-learn versions.
    
    Args:
        model_path (str): Path to the pickled model file
        
    Returns:
        The loaded model or None if loading fails
    """
    # If model_path is None, try to find the model file
    if model_path is None:
        return None
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Get file size and check if it's valid
    try:
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size} bytes")
        if file_size == 0:
            print(f"Error: Model file {model_path} is empty")
            return None
    except Exception as e:
        print(f"Error checking model file: {str(e)}")
    
    # Suppress specific warnings about node array dtype
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            
            # Try with a different approach if the first one fails
            try:
                import joblib
                print(f"Attempting to load with joblib...")
                model = joblib.load(model_path)
                print(f"Successfully loaded model with joblib from {model_path}")
                return model
            except Exception as e2:
                print(f"Error loading model with joblib: {str(e2)}")
                return None

def is_model_compatible(model):
    """
    Check if a loaded model is compatible with the current scikit-learn version.
    
    Args:
        model: The loaded model to check
        
    Returns:
        bool: True if the model is compatible, False otherwise
    """
    if model is None:
        return False
    
    # Check if it's a scikit-learn estimator
    if not isinstance(model, BaseEstimator):
        return True  # Not a scikit-learn model, so no compatibility issues
    
    # Try a simple operation that would fail if the model is incompatible
    try:
        if hasattr(model, 'feature_importances_'):
            _ = model.feature_importances_
        return True
    except:
        return False