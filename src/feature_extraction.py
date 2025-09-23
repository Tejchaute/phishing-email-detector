import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def extract_features(input_csv=r"data\merged_cleaned_emails.csv", 
                     input_df=None,
                     output_csv=r"data\features.csv", 
                     vectorizer_pkl=r"models\vectorizer.pkl",
                     vectorizer=None,
                     save_feature_columns=True,
                     feature_columns_pkl=r"models/feature_columns.pkl"):

    # Load DataFrame
    if input_df is not None:
        df = input_df.copy()
    elif input_csv is not None:
        df = pd.read_csv(input_csv)
    else:
        raise ValueError("Provide either input_csv or input_df")

    # =====================
    # Handcrafted Features
    # =====================
    df['subject_length'] = df['subject'].apply(lambda x: len(str(x)))
    df['body_length'] = df['body'].apply(lambda x: len(str(x)))
    df['subject_exclamations'] = df['subject'].apply(lambda x: str(x).count('!'))
    df['body_exclamations'] = df['body'].apply(lambda x: str(x).count('!'))

    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['body_url_count'] = df['body'].apply(lambda x: len(re.findall(url_pattern, str(x))))

    suspicious_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
    df['sender_suspicious'] = df['sender'].apply(
        lambda x: 1 if any(domain in str(x).lower() for domain in suspicious_domains) else 0
    )

    # =====================
    # New Features
    # =====================
    df['has_html'] = df['body'].apply(lambda x: 1 if bool(re.search(r"<[^>]+>", str(x))) else 0)

    suspicious_keywords = ['verify', 'account', 'login', 'bank', 'password', 'urgent', 'click', 'confirm']
    df['suspicious_keyword_count'] = df['body'].apply(
        lambda text: sum(word in str(text).lower() for word in suspicious_keywords)
    )

    shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co']
    df['shortened_url_count'] = df['body'].apply(
        lambda text: sum(service in str(text).lower() for service in shortening_services)
    )

    trusted_domains = ['microsoft.com', 'paypal.com', 'amazon.com', 'apple.com']
    df['sender_trusted'] = df['sender'].apply(
        lambda x: 1 if any(domain in str(x).lower() for domain in trusted_domains) else 0
    )

    # =====================
    # Encode categorical features
    # =====================
    le_sender = LabelEncoder()
    df['sender_encoded'] = le_sender.fit_transform(df['sender'].astype(str))

    le_receiver = LabelEncoder()
    df['receiver_encoded'] = le_receiver.fit_transform(df['receiver'].astype(str))

    # =====================
    # Text Vectorization (TF-IDF)
    # =====================
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=500)
        text_features = vectorizer.fit_transform(df['subject'].astype(str) + " " + df['body'].astype(str))
        # Save vectorizer for future inference
        if vectorizer_pkl:
            with open(vectorizer_pkl, "wb") as f:
                pickle.dump(vectorizer, f)
            print(f"[INFO] Vectorizer saved to {vectorizer_pkl}")
    else:
        text_features = vectorizer.transform(df['subject'].astype(str) + " " + df['body'].astype(str))

    tfidf_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

    # =====================
    # Combine Features
    # =====================
    base_columns = [
        'label', 'urls', 'subject_length', 'body_length',
        'subject_exclamations', 'body_exclamations',
        'body_url_count', 'sender_suspicious',
        'has_html', 'suspicious_keyword_count', 'shortened_url_count',
        'sender_trusted', 'sender_encoded', 'receiver_encoded'
    ]
    base_columns = [col for col in base_columns if col in df.columns]

    feature_df = pd.concat([df[base_columns].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # Save feature_columns.pkl for future inference
    if save_feature_columns:
        feature_columns = feature_df.drop(columns=['label'], errors='ignore').columns.tolist()
        with open(feature_columns_pkl, "wb") as f:
            pickle.dump(feature_columns, f)
        print(f"[INFO] Feature columns saved to {feature_columns_pkl}")


    # Save features for full dataset only
    if output_csv and input_df is None:
        feature_df.to_csv(output_csv, index=False)
        print(f"[INFO] Features saved to {output_csv}")

    return feature_df


if __name__ == "__main__":
    features = extract_features()
    print(features.head())
