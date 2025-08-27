def log_message(message):
    print(f"[LOG] {message}")

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)
    log_message(f"Model saved to {filename}")

def load_model(filename):
    import joblib
    model = joblib.load(filename)
    log_message(f"Model loaded from {filename}")
    return model

def visualize_data(data, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(data.keys(), data.values())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()