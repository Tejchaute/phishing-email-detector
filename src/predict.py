import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================
# Load trained artifacts
# ==========================
with open(r"models/vectorizer.pkl", "rb") as f:
    vectorizer: TfidfVectorizer = pickle.load(f)

with open(r"models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open(r"models/randomforest_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================
# Helper functions
# ==========================
def build_features(subject: str, body: str, sender: str):
    """Extract features from raw email fields"""

    # Handcrafted features
    features = {
        "subject_length": len(subject),
        "body_length": len(body),
        "subject_exclamations": subject.count("!"),
        "body_exclamations": body.count("!"),
        "body_url_count": len(re.findall(r"http[s]?://\S+", body)),
        "has_html": 1 if re.search(r"<[^>]+>", body) else 0,
    }

    # Sender checks
    suspicious_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
    trusted_domains = ["microsoft.com", "paypal.com", "amazon.com", "apple.com"]

    features["sender_suspicious"] = 1 if any(dom in sender.lower() for dom in suspicious_domains) else 0
    features["sender_trusted"] = 1 if any(dom in sender.lower() for dom in trusted_domains) else 0

    # Keyword features
    suspicious_keywords = [
        "verify", "account", "login", "bank", "password",
        "urgent", "click", "confirm"
    ]
    features["suspicious_keyword_count"] = sum(word in body.lower() for word in suspicious_keywords)

    shortening_services = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co"]
    features["shortened_url_count"] = sum(service in body.lower() for service in shortening_services)

    # Encode sender (fallback: hash trick)
    features["sender_encoded"] = abs(hash(sender)) % (10**6)

    # Receiver not available → set dummy
    features["receiver_encoded"] = 0

    # ==========================
    # TF-IDF Features
    # ==========================
    text = subject + " " + body
    tfidf_vector = vectorizer.transform([text])
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names_out())

    # ==========================
    # Merge all features
    # ==========================
    features_df = pd.DataFrame([features])
    final_df = pd.concat([features_df, tfidf_df], axis=1)

    # Reindex columns to match training features
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    return final_df

# ==========================
# Prediction function
# ==========================
def predict_email(subject: str, body: str, sender: str):
    features = build_features(subject, body, sender)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    if prediction == 1:
        print(f"\n🚨 This email is likely PHISHING (confidence: {prob:.2f})")
    else:
        print(f"\n✅ This email looks LEGIT (confidence: {prob:.2f})")

# ==========================
# Main execution
# ==========================
if __name__ == "__main__":
    subject = input("Enter email subject: ")
    body = input("Enter email body: ")
    sender = input("Enter sender email address: ")

    predict_email(subject, body, sender)
