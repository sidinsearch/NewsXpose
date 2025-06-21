import os
import sys
import warnings

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the safe_load_model function
from model_utils import safe_load_model, is_model_compatible

def test_model_loading():
    """Test loading models with the safe_load_model function"""
    print("Testing model loading...")
    
    # Test ensemble model
    print("\nTesting ensemble model:")
    ensemble_path = 'ensemble_fake_news_detector.pkl'
    if os.path.exists(ensemble_path):
        ensemble_data = safe_load_model(ensemble_path)
        if ensemble_data:
            ensemble_model, vector = ensemble_data
            print(f"✅ Successfully loaded ensemble model")
            print(f"   Model type: {type(ensemble_model).__name__}")
            print(f"   Vectorizer type: {type(vector).__name__}")
            print(f"   Model compatible: {is_model_compatible(ensemble_model)}")
        else:
            print(f"❌ Failed to load ensemble model")
    else:
        print(f"⚠️ Ensemble model file not found: {ensemble_path}")
    
    # Test image model
    print("\nTesting image model:")
    image_model_path = 'image-model.pkl'
    if os.path.exists(image_model_path):
        image_model = safe_load_model(image_model_path)
        if image_model:
            print(f"✅ Successfully loaded image model")
            print(f"   Model type: {type(image_model).__name__}")
            print(f"   Model compatible: {is_model_compatible(image_model)}")
        else:
            print(f"❌ Failed to load image model")
    else:
        print(f"⚠️ Image model file not found: {image_model_path}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    # Suppress warnings during testing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_model_loading()