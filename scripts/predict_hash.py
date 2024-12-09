import joblib
import math
import pandas as pd
import os

# Load the trained model and label encoder
def load_model_and_encoder(model_filename, encoder_filename):
    model = joblib.load(model_filename)
    label_encoder = joblib.load(encoder_filename)
    return model, label_encoder

# Helper function to analyze hash characteristics
def analyze_hash(hash_value):
    length = len(hash_value)
    contains_numbers = any(char.isdigit() for char in hash_value)
    contains_letters = any(char.isalpha() for char in hash_value)
    contains_special = any(not char.isalnum() for char in hash_value)
    entropy = -sum((hash_value.count(c) / len(hash_value)) * math.log2(hash_value.count(c) / len(hash_value)) for c in set(hash_value))

    return {
        "Length": length,
        "Contains Numbers": contains_numbers,
        "Contains Letters": contains_letters,
        "Contains Special Characters": contains_special,
        "Entropy": entropy
    }

# Predict the Hash Algorithm
def predict_hash_algorithm(model, label_encoder, hash_value):
    # Extract features from the hash value
    features = analyze_hash(hash_value)

    # Prepare feature vector with feature names
    feature_vector = pd.DataFrame([{
        "Length": features["Length"],
        "Contains Numbers": features["Contains Numbers"],
        "Contains Letters": features["Contains Letters"],
        "Contains Special Characters": features["Contains Special Characters"],
        "Entropy": features["Entropy"]
    }])

    # Make the prediction
    prediction = model.predict(feature_vector)

    # Decode the prediction to get the hash algorithm
    hash_algorithm = label_encoder.inverse_transform(prediction)
    return hash_algorithm[0]

# Main Execution
if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the trained model and label encoder
    model_filename =  os.path.join(data_dir, '../models/hash_algorithm_predictor.pkl')
    encoder_filename =  os.path.join(data_dir, '../models/label_encoder.pkl')
    model, label_encoder = load_model_and_encoder(model_filename, encoder_filename)

    # Input hash value for prediction
    sample_hash = input("Enter the hash value to predict the algorithm: ")

    # Predict the algorithm
    predicted_algorithm = predict_hash_algorithm(model, label_encoder, sample_hash)
    print(f"The predicted hash algorithm for the given hash is: {predicted_algorithm}")
