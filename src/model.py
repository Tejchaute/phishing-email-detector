from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

class PhishingEmailDetector:
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data
    
    def preprocess_data(self, data):
        # Assuming the preprocessing steps are handled in preprocessing.py
        # This function should call the relevant preprocessing functions
        return data
    
    def extract_features(self, data):
        # Assuming feature extraction is handled in feature_extraction.py
        # This function should call the relevant feature extraction functions
        return data
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        print(classification_report(y, predictions))
        print(f'Accuracy: {accuracy_score(y, predictions)}')
    
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
    
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
    
    def predict(self, X):
        return self.model.predict(X)