import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(
    features_csv=r"data\features.csv",
    model_pkl=r"models\randomforest_model.pkl"
):
    # Load features
    print("[INFO] Loading features...")
    df = pd.read_csv(features_csv)

    # Separate features & labels
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")

    X = df.drop(columns=['label'])
    y = df['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # Train
    print("[INFO] Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    with open(model_pkl, "wb") as f:
        pickle.dump(clf, f)
    print(f"[INFO] Model saved to {model_pkl}")

    return clf

if __name__ == "__main__":
    trained_model = train_model()
