import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def analyze_feature_importance(features_csv=r"data\features.csv"):
    # Load feature dataset
    df = pd.read_csv(features_csv)

    # Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n[INFO] Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Show top 20 features
    print("\n[INFO] Top 20 Important Features:\n")
    print(importance_df.head(20))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_df.head(20).plot(
        kind='barh', x='Feature', y='Importance', legend=False
    )
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    return importance_df


if __name__ == "__main__":
    importance_df = analyze_feature_importance()
