import pickle

from model_utils import safe_load_model, is_model_compatible

# Load the existing models
with open('ensemble_fake_news_detector.pkl', 'rb') as f:
    ensemble_model, vector = pickle.load(f)

with open('image-model.pkl', 'rb') as f:
    image_model = pickle.load(f)

# Combine into a single dictionary
combined_models = {
    'ensemble_model': ensemble_model,
    'vectorizer': vector,
    'image_model': image_model
}

# Save the combined dictionary into one .pkl file
with open('combined_models.pkl', 'wb') as f:
    pickle.dump(combined_models, f)

print("Combined model saved as 'combined_models.pkl'")
