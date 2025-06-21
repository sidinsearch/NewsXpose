import os
import pickle
import warnings
import numpy as np
from sklearn.base import BaseEstimator

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
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Suppress specific warnings about node array dtype
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
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