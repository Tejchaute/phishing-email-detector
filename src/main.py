import pandas as pd
from preprocessing import preprocess_data
from feature_extraction import extract_features
from model import train_model, load_model
from predict import make_predictions

def main():
    # Load the dataset
    data = pd.read_csv('data/Nazario_5.csv')
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Extract features
    X, y = extract_features(processed_data)
    
    # Train the model
    model = train_model(X, y)
    
    # Save the model for future predictions
    model.save('model/phishing_detector_model.pkl')
    
    # Load the model for predictions
    loaded_model = load_model('model/phishing_detector_model.pkl')
    
    # Example of making predictions on new data
    # new_data = pd.read_csv('data/new_emails.csv')
    # predictions = make_predictions(loaded_model, new_data)
    # print(predictions)

if __name__ == "__main__":
    main()