# main.py

import sys
from preprocessing import merge_and_clean_csv_files
from feature_extraction import extract_features
from model import train_model
from predict import predict_email

# ==========================
# Paths (central config)
# ==========================
RAW_FILES = [
    r"data/CEAS_08.csv",
    r"data/Nigerian_Fraud.csv",
    r"data/SpamAssasin.csv",
    r"data/Nazario.csv",
]
CLEANED_FILE = r"data/merged_cleaned_emails.csv"
FEATURES_FILE = r"data/features.csv"
MODEL_FILE = r"models/randomforest_model.pkl"


# ==========================
# Menu System
# ==========================
def main():
    print("\n=== Phishing Email Detector ===")
    print("1. Preprocess dataset")
    print("2. Extract features")
    print("3. Train model")
    print("4. Predict email")
    print("5. Exit")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        merge_and_clean_csv_files(RAW_FILES, CLEANED_FILE)

    elif choice == "2":
        extract_features(input_csv=CLEANED_FILE, output_csv=FEATURES_FILE)

    elif choice == "3":
        train_model(features_csv=FEATURES_FILE, model_pkl=MODEL_FILE)

    elif choice == "4":
        subject = input("Enter email subject: ")
        body = input("Enter email body: ")
        sender = input("Enter sender email: ")
        predict_email(subject, body, sender)

    elif choice == "5":
        print("Exiting...")
        sys.exit(0)

    else:
        print("[ERROR] Invalid choice, please select again.")


if __name__ == "__main__":
    while True:
        main()
