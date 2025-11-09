# language_detection.py (Enhanced Version)

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version inconsistency warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# 1. Load dataset
print("Loading dataset...")
data = pd.read_csv("Language Detection.csv")
print(f"Dataset loaded: {len(data)} samples")
print(data.head())

# 2. Keep top 6 languages for better accuracy
top_languages = ['English', 'French', 'Spanish', 'German', 'Italian', 'Hindi']
data = data[data['Language'].isin(top_languages)].reset_index(drop=True)
print(f"Filtered to {len(data)} samples in top 6 languages")
print(f"Language distribution:\n{data['Language'].value_counts()}\n")

# 3. Clean text
def clean_text(text):
    text = re.sub(r'[^A-Za-z\u0900-\u097F\s]', '', str(text))  # keep Hindi characters
    text = text.lower().strip()
    return text

data['Text'] = data['Text'].apply(clean_text)
data = data[data['Text'].str.len() > 0]  # Remove empty texts
print(f"Text cleaned. {len(data)} samples remaining\n")

# 4. Feature extraction using TF-IDF (char n-grams 2-4)
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
X = vectorizer.fit_transform(data['Text'])
y = data['Language']
print(f"Features extracted: {X.shape[1]} features from {X.shape[0]} samples\n")

# 5. Split data into train/test (stratified)
print("Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}\n")

# 6. Train multiple models
print("Training models...\n")

# Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Logistic Regression
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# 7. Evaluate models
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

models = {'Naive Bayes': model_nb, 'Logistic Regression': model_lr}

best_model = None
best_accuracy = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")

# 8. Generate confusion matrix
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_languages, yticklabels=top_languages)
plt.title(f'{best_model_name} - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# 9. Save best model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(best_model, "language_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved successfully!\n")

# 10. Test with sample sentences
print("=" * 60)
print("TESTING WITH SAMPLE TEXTS")
print("=" * 60)

sample_texts = [
    "Bonjour mon ami, comment allez-vous?",          # French
    "Main tumhe kal subah mil jaata hoon",           # Hindi
    "I love natural language processing and AI",     # English
    "Hola, como estas? Espero que bien",           # Spanish
    "Ich liebe Deutsch und Mathematik",            # German
    "Ciao, come stai? Ti amo molto"                # Italian
]

sample_features = vectorizer.transform(sample_texts)
predicted_languages = best_model.predict(sample_features)
probabilities = best_model.predict_proba(sample_features)

print()
for text, lang, probs in zip(sample_texts, predicted_languages, probabilities):
    max_prob = np.max(probs) * 100
    print(f"Text: '{text}'")
    print(f"  -> Detected: {lang} ({max_prob:.1f}% confidence)\n")