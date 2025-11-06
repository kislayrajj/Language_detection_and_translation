import joblib

# Load saved model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Test new input
text = input("Enter text: ")
features = vectorizer.transform([text])
prediction = model.predict(features)[0]

print(f"Detected Language: {prediction}")
