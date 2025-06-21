import os
import sys
import warnings
import pickle
import numpy as np
from sklearn.base import BaseEstimator

def test_model_loading():
    """Test loading models with warning suppression"""
    print("Testing model loading with warning suppression...")
    
    # Suppress the specific warning
    warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")
    
    # Test ensemble model
    print("\nTesting ensemble model:")
    ensemble_path = 'ensemble_fake_news_detector.pkl'
    if os.path.exists(ensemble_path):
        try:
            with open(ensemble_path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            if ensemble_data:
                ensemble_model, vector = ensemble_data
                print(f"✅ Successfully loaded ensemble model")
                print(f"   Model type: {type(ensemble_model).__name__}")
                print(f"   Vectorizer type: {type(vector).__name__}")
                
                # Test if the model can be used for prediction
                try:
                    # Create a simple test input
                    test_text = "This is a test article for fake news detection"
                    test_vector = vector.transform([test_text])
                    prediction = ensemble_model.predict(test_vector)
                    print(f"   Test prediction successful: {prediction}")
                except Exception as e:
                    print(f"❌ Test prediction failed: {str(e)}")
            else:
                print(f"❌ Failed to load ensemble model")
        except Exception as e:
            print(f"❌ Error loading ensemble model: {str(e)}")
    else:
        print(f"⚠️ Ensemble model file not found: {ensemble_path}")
    
    # Test image model
    print("\nTesting image model:")
    image_model_path = 'image-model.pkl'
    if os.path.exists(image_model_path):
        try:
            with open(image_model_path, 'rb') as f:
                image_model = pickle.load(f)
            
            if image_model:
                print(f"✅ Successfully loaded image model")
                print(f"   Model type: {type(image_model).__name__}")
            else:
                print(f"❌ Failed to load image model")
        except Exception as e:
            print(f"❌ Error loading image model: {str(e)}")
    else:
        print(f"⚠️ Image model file not found: {image_model_path}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_model_loading()