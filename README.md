# 📧 Phishing Email Detector

A Machine Learning and NLP-based project to detect phishing emails using classical ML algorithms and text feature extraction.  
The project preprocesses raw email datasets, extracts features with TF-IDF, trains a RandomForest classifier, and provides predictions on new email inputs.

---

## 📂 Project Structure

PHISHING-EMAIL-DETECTOR/
│
├── data/ # Raw and preprocessed datasets
│ ├── CEAS_08.csv
│ ├── Nazario.csv
│ ├── Nigerian_Fraud.csv
│ ├── SpamAssasin.csv
│ ├── merged_cleaned_emails.csv
│ └── features.csv
│
├── models/ # Saved models and encoders
│ ├── best_model.pkl
│ ├── randomForest_model.pkl
│ ├── vectorizer.pkl
│ └── feature_columns.pkl
│
├── src/ # Source code
│ ├── preprocessing.py # Cleans and preprocesses raw email dataset
│ ├── feature_extraction.py # Extracts features using TF-IDF & LabelEncoder
│ ├── model.py # Trains RandomForestClassifier on features
│ ├── predict.py # Predicts if an email is phishing or not
│ ├── feature_importance.py # Visualizes / analyzes feature importance
│ └── main.py # Entry point to run the pipeline end-to-end
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/your-username/phishing-email-detector.git
   cd phishing-email-detector
2. Create a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
3. Install dependencies:
   pip install -r requirements.txt


🚀 Usage
1. Preprocess the dataset
   python src/preprocessing.py
Generates merged_cleaned_emails.csv with cleaned emails.

2. Extract features
   python src/feature_extraction.py
Creates TF-IDF features and encodes labels.

3. Train the model
   python src/model.py
Trains a RandomForestClassifier and saves the model to models/.

4. Predict on new input
   python src/predict.py
Enter subject, body, and sender email address when prompted, and get a phishing prediction.

5. Run full pipeline
   python src/main.py
Runs preprocessing → feature extraction → model training → prediction.

📊 Example Predictions
Phishing email input:
   Sender: security@fakebank.com
   Subject: Urgent! Verify your account
   Body: Your account has been locked. Click here to verify immediately: http://fakeurl.com
➡️ Output: Phishing Detected

Legitimate email input:
   Sender: hr@company.com
   Subject: Meeting Reminder
   Body: This is a reminder for the scheduled team meeting tomorrow at 10 AM.
➡️ Output: Not Phishing


📌 Features
   Preprocessing of raw datasets

   Feature extraction with TF-IDF

   Classification using RandomForest

   Custom prediction on user input

   Feature importance visualization


📚 Requirements
   Python 3.9+

   pandas

   scikit-learn

(Install with pip install -r requirements.txt)


👨‍💻 Author
Developed by Tej chaute
If you like this project, consider ⭐ starring the repo!
