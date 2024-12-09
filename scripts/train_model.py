import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load JSON Data
def load_json_data(input_filename):
    with open(input_filename, 'r') as f:
        data = json.load(f)
    return data

# Prepare Data for Model Training
def prepare_data_from_json(data):
    dataset = []

    # Process each entry in the dataset
    for entry in data:
        hash_value = entry['Hash Value']
        hash_algorithm = entry['Hash Algorithm']
        features = {
            'Length': entry['Length'],
            'Contains Numbers': entry['Contains Numbers'],
            'Contains Letters': entry['Contains Letters'],
            'Contains Special Characters': entry['Contains Special Characters'],
            'Entropy': entry['Entropy']
        }
        
        dataset.append({
            "Input Value": entry["Input Value"],
            "Hash Algorithm": hash_algorithm,
            "Hash Value": hash_value,
            **features,
        })

    return dataset

# Train the Model
def train_model(dataset):
    df = pd.DataFrame(dataset)

    # Select features and labels
    X = df[['Length', 'Contains Numbers', 'Contains Letters', 'Contains Special Characters', 'Entropy']]
    y = df['Hash Algorithm']

    # Encode labels (Hash Algorithm) to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Define the directory where you want to save the files
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")

    # Construct the full file paths
    model_path = os.path.join(save_dir, "hash_algorithm_predictor.pkl")
    label_encoder_path = os.path.join(save_dir, "label_encoder.pkl")

    # Save the model and the label encoder for later use
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_encoder_path)

    return model, label_encoder

# Main Execution
if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the dataset from the JSON file
    input_filename =  os.path.join(data_dir,"../data/hash_dataset.json")  # Input JSON file
    data = load_json_data(input_filename)

    # Prepare the dataset for training
    dataset = prepare_data_from_json(data)

    # Train the model on the dataset
    model, label_encoder = train_model(dataset)

