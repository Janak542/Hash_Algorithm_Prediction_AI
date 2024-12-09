# Hash Algorithm Prediction AI

A Python-based machine learning project to predict the cryptographic hash algorithm (e.g., MD5, SHA1, SHA256) used to generate a given hash value. The model analyzes various characteristics of the hash and predicts the most likely algorithm.

---

## Features

- Supports prediction for common hash algorithms:
  - MD5
  - SHA1
  - SHA256
- Custom dataset generation from input strings.
- AI model using Random Forest Classifier for predictions.
- Includes entropy calculation and other hash-specific features.

---

## File Structure

```plaintext
hash-prediction-ai/
├── data/
│   └── hash_dataset.json         # JSON dataset file
├── models/
│   ├── hash_algorithm_predictor.pkl  # Trained model
│   └── label_encoder.pkl         # Label encoder
├── scripts/
│   ├── generate_dataset.py       # Script to generate the JSON dataset
│   ├── train_model.py            # Script to train the AI model
│   └── predict_hash.py           # Script to use the model for prediction
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation for the project
└── .gitignore                    # Files to ignore in the repository



# Used These Algorithms for AI Model

MD5
SHA-1
SHA-256
