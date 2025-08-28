# Phishing Email Detector

This project implements an advanced phishing email detection system using machine learning techniques. The model is trained on a dataset containing both phishing and non-phishing emails, utilizing various features extracted from the email content.

## Project Structure

```
phishing-email-detector
├── data
│   └── merged_cleaned_emails.csv          # Dataset containing email data with labels
├── src
│   ├── main.py                # Entry point of the application
│   ├── preprocessing.py       # Data cleaning and preprocessing functions
│   ├── feature_extraction.py  # Functions for feature extraction
|   ├── feature_importance.py  # Functions for measure importance of features
│   ├── model.py               # Machine learning model definition and training
│   ├── predict.py             # Functions for making predictions
│   └── utils.py               # Utility functions used across modules
├── requirements.txt           # List of project dependencies
└── README.md                  # Project documentation
```

## Dataset

The dataset used for this project is located in the `data` directory as `merged_cleaned_emails.csv`. It includes the following features:
- **Sender**: The email address of the sender.
- **Receiver**: The email address of the receiver.
- **Date**: The date the email was sent.
- **Subject**: The subject line of the email.
- **Body**: The content of the email.
- **URLs**: Any URLs present in the email.
- **Label**: Indicates whether the email is phishing (1) or non-phishing (0).

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd phishing-email-detector
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the phishing email detection system, execute the following command:
```
python src/main.py
```

This will initiate the workflow of loading the dataset, preprocessing the data, extracting features, training the model, and making predictions.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.