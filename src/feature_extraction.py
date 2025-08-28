# feature_extraction.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def extract_features(input_csv=r"data\merged_cleaned_emails.csv", output_csv=r"data\features.csv"):
    # Load dataset
    df = pd.read_csv(input_csv)

    # =====================
    # Handcrafted Features
    # =====================

    # Length-based features
    df['subject_length'] = df['subject'].apply(lambda x: len(str(x)))
    df['body_length'] = df['body'].apply(lambda x: len(str(x)))

    # Count of special characters in subject & body
    df['subject_exclamations'] = df['subject'].apply(lambda x: str(x).count('!'))
    df['body_exclamations'] = df['body'].apply(lambda x: str(x).count('!'))

    # Count of URLs in body text
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['body_url_count'] = df['body'].apply(lambda x: len(re.findall(url_pattern, str(x))))

    # Whether sender looks suspicious (free email domains etc.)
    suspicious_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
    df['sender_suspicious'] = df['sender'].apply(
        lambda x: 1 if any(domain in str(x).lower() for domain in suspicious_domains) else 0
    )

    # =====================
    # New Features
    # =====================

    # Detect HTML content in email body
    df['has_html'] = df['body'].apply(lambda x: 1 if bool(re.search(r"<[^>]+>", str(x))) else 0)

    # Suspicious keyword frequency
    suspicious_keywords = ['verify', 'account', 'login', 'bank', 'password', 'urgent', 'click', 'confirm']
    df['suspicious_keyword_count'] = df['body'].apply(
        lambda text: sum(word in str(text).lower() for word in suspicious_keywords)
    )

    # Detect shortened URLs (bit.ly, tinyurl, etc.)
    shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co']
    df['shortened_url_count'] = df['body'].apply(
        lambda text: sum(service in str(text).lower() for service in shortening_services)
    )

    # Sender domain reputation check (basic)
    trusted_domains = ['microsoft.com', 'paypal.com', 'amazon.com', 'apple.com']
    df['sender_trusted'] = df['sender'].apply(
        lambda x: 1 if any(domain in str(x).lower() for domain in trusted_domains) else 0
    )

    # =====================
    # Encoding categorical features
    # =====================
    le_sender = LabelEncoder()
    df['sender_encoded'] = le_sender.fit_transform(df['sender'].astype(str))

    le_receiver = LabelEncoder()
    df['receiver_encoded'] = le_receiver.fit_transform(df['receiver'].astype(str))

    # =====================
    # Text Vectorization (TF-IDF on subject + body)
    # =====================
    vectorizer = TfidfVectorizer(max_features=500)
    text_features = vectorizer.fit_transform(df['subject'].astype(str) + " " + df['body'].astype(str))
    tfidf_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

    # =====================
    # Combine Features
    # =====================
    feature_df = pd.concat([
        df[['label', 'urls', 'subject_length', 'body_length',
            'subject_exclamations', 'body_exclamations',
            'body_url_count', 'sender_suspicious',
            'has_html', 'suspicious_keyword_count', 'shortened_url_count',
            'sender_trusted', 'has_attachment',
            'sender_encoded', 'receiver_encoded'
        ]].reset_index(drop=True),
        tfidf_df.reset_index(drop=True)
    ], axis=1)

    # Save features
    feature_df.to_csv(output_csv, index=False)
    print(f"[INFO] Features saved to {output_csv}")
    return feature_df


if __name__ == "__main__":
    features = extract_features()
    print(features.head())
