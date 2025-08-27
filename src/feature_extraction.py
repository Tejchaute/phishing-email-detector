import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(data):
    # Extracting features from the email body using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['body'])
    
    # Converting categorical features to numerical format
    data['sender'] = data['sender'].astype('category').cat.codes
    data['receiver'] = data['receiver'].astype('category').cat.codes
    data['subject'] = data['subject'].astype('category').cat.codes
    
    # Combining all features into a single DataFrame
    features = pd.concat([data[['sender', 'receiver', 'subject']], 
                          pd.DataFrame(tfidf_matrix.toarray())], axis=1)
    
    return features, data['label']