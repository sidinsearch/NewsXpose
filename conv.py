import pickle
import joblib
# Convert ensemble model and vectorizer
with open('ensemble_fake_news_detector.pkl', 'rb') as f:
    ensemble_model, vector = pickle.load(f)
joblib.dump((ensemble_model, vector), 'ensemble_fake_news_detector.joblib')
# For image model (if needed)
with open('image-model.pkl', 'rb') as f:
    image_model = pickle.load(f)
joblib.dump(image_model, 'image-model.joblib')