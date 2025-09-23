📧 Phishing Email Detector

A machine learning project to detect phishing emails using text preprocessing, feature extraction, and classification models. The system leverages datasets of spam and phishing emails, extracts linguistic and metadata-based features, and predicts whether an email is phishing or legitimate.

📂 Project Structure
Phishing-Email-Detector/
│
├── data/                        # Datasets used for training/testing
│   ├── CEAS_08.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   ├── SpamAssasin.csv
│   ├── merged_cleaned_emails.csv
│   └── features.csv
│
├── models/                      # Saved models & preprocessing files
│   ├── best_model.pkl
│   ├── randomForest_model.pkl
│   ├── vectorizer.pkl
│   └── feature_columns.pkl
│
├── src/                         # Source code
│   ├── feature_extraction.py    # Extracts features from raw emails
│   ├── feature_importance.py    # Analyzes feature importance
│   ├── preprocessing.py         # Data cleaning & preprocessing
│   ├── model.py                 # Model training & saving
│   ├── predict.py               # Prediction script
│   └── main.py                  # Entry point script
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitattributes               # Git configuration

⚙️ Installation

1. Clone this repository:

   git clone https://github.com/Tejchaute/phishing-email-detector
   cd phishing-email-detector


2. Create a virtual environment:

   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows


3. Install dependencies:

   pip install -r requirements.txt

🚀 Usage
1. Train the Model

Run the training pipeline (preprocessing, feature extraction, training):

   python src/model.py

2. Predict Phishing Emails

You can test the model interactively:

   python src/predict.py


Example input:

   === Email Phishing Detector ===
   Enter email subject: Urgent! Verify your account
   Enter email body: Your account has been suspended. Click here to confirm.
   Enter sender email: support@paypal.com


Output:

   Prediction: 🚨 Phishing Email Detected!

3. Analyze Feature Importance
   python src/feature_importance.py

📊 Datasets Used

CEAS 2008 Spam Filter Challenge Dataset

Nazario Phishing Corpus

Nigerian Fraud Emails Dataset

SpamAssassin Corpus

Custom merged dataset (merged_cleaned_emails.csv)

🧠 Models

Random Forest Classifier (primary model)

Vectorization via TF-IDF

Best model saved as best_model.pkl

📌 Features Extracted

Text-based features (word counts, special characters, suspicious keywords)

Email header features (sender address, domain)

Statistical & NLP-based features

✅ Future Improvements

Add deep learning models (LSTM, BERT)

Deploy as a web API or Flask app

Real-time email classification

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License.