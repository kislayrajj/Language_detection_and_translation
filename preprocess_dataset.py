import pandas as pd
import numpy as np
from collections import Counter
import re
import os

def load_wili_data(data_dir='data'):
    """Load WILI-2018 dataset files"""
    print("Loading WILI-2018 dataset...")

    # Load labels
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'), sep=';')
    label_to_lang = dict(zip(labels_df['Label'], labels_df['English']))
    lang_to_label = dict(zip(labels_df['English'], labels_df['Label']))

    # Load training data
    with open(os.path.join(data_dir, 'x_train.txt'), 'r', encoding='utf-8') as f:
        x_train = f.read().splitlines()

    with open(os.path.join(data_dir, 'y_train.txt'), 'r', encoding='utf-8') as f:
        y_train = f.read().splitlines()

    # Load test data
    with open(os.path.join(data_dir, 'x_test.txt'), 'r', encoding='utf-8') as f:
        x_test = f.read().splitlines()

    with open(os.path.join(data_dir, 'y_test.txt'), 'r', encoding='utf-8') as f:
        y_test = f.read().splitlines()

    # Convert labels to language names
    y_train_lang = [label_to_lang.get(label, label) for label in y_train]
    y_test_lang = [label_to_lang.get(label, label) for label in y_test]

    print(f"Train set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")

    return x_train, y_train_lang, x_test, y_test_lang, label_to_lang

def get_top_50_languages(y_train, y_test):
    """Get the 50 most spoken languages based on training data"""
    all_labels = y_train + y_test
    label_counts = Counter(all_labels)

    # Top 50 most frequent languages
    top_50 = [lang for lang, _ in label_counts.most_common(50)]
    print(f"Top 50 languages: {top_50[:10]}...")  # Show first 10

    return top_50

def filter_by_languages(x_data, y_data, target_languages):
    """Filter data to only include target languages"""
    filtered_x = []
    filtered_y = []

    for text, lang in zip(x_data, y_data):
        if lang in target_languages:
            filtered_x.append(text)
            filtered_y.append(lang)

    return filtered_x, filtered_y

def clean_text(text):
    """Clean and normalize text data"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', str(text).strip())

    # Remove non-printable characters but keep multilingual characters
    text = re.sub(r'[^\x20-\x7E\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF\u10000-\u1FFFD\u20000-\u2FFFD\u30000-\u3FFFD\u40000-\u4FFFD\u50000-\u5FFFD\u60000-\u6FFFD\u70000-\u7FFFD\u80000-\u8FFFD\u90000-\u9FFFD\uA0000-\uAFFFD\uB0000-\uBFFFD\uC0000-\uCFFFD\uD0000-\uDFFFD\uE0000-\uEFFFD]+', '', text)

    # Normalize quotes and apostrophes
    text = re.sub(r'[""''""]', '"', text)
    text = re.sub(r'[''`]', "'", text)

    return text.strip()

def balance_dataset(x_data, y_data, min_samples=10000):
    """Balance dataset by undersampling to minimum samples per language"""
    from collections import defaultdict

    # Group by language
    lang_groups = defaultdict(list)
    for text, lang in zip(x_data, y_data):
        lang_groups[lang].append(text)

    balanced_x = []
    balanced_y = []

    for lang, texts in lang_groups.items():
        # Take minimum of available samples and min_samples
        n_samples = min(len(texts), min_samples)
        selected_texts = np.random.choice(texts, n_samples, replace=False)

        balanced_x.extend(selected_texts)
        balanced_y.extend([lang] * n_samples)

    return balanced_x, balanced_y

def create_splits(x_data, y_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Create train/validation/test splits"""
    from sklearn.model_selection import train_test_split

    # First split: separate test set
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_data, y_data, test_size=test_ratio, random_state=random_state, stratify=y_data
    )

    # Second split: separate validation from remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_temp
    )

    return x_train, y_train, x_val, y_val, x_test, y_test

def main():
    # Load data
    x_train, y_train, x_test, y_test, label_to_lang = load_wili_data()

    # Get top 50 languages
    top_50_languages = get_top_50_languages(y_train, y_test)

    # Filter data
    x_train_filtered, y_train_filtered = filter_by_languages(x_train, y_train, top_50_languages)
    x_test_filtered, y_test_filtered = filter_by_languages(x_test, y_test, top_50_languages)

    print(f"After filtering to top 50 languages:")
    print(f"Train set: {len(x_train_filtered)} samples")
    print(f"Test set: {len(x_test_filtered)} samples")

    # Clean text
    print("Cleaning text data...")
    x_train_cleaned = [clean_text(text) for text in x_train_filtered]
    x_test_cleaned = [clean_text(text) for text in x_test_filtered]

    # Remove empty texts
    train_data = [(text, lang) for text, lang in zip(x_train_cleaned, y_train_filtered) if text.strip()]
    test_data = [(text, lang) for text, lang in zip(x_test_cleaned, y_test_filtered) if text.strip()]

    x_train_cleaned, y_train_filtered = zip(*train_data)
    x_test_cleaned, y_test_filtered = zip(*test_data)

    print(f"After cleaning:")
    print(f"Train set: {len(x_train_cleaned)} samples")
    print(f"Test set: {len(x_test_cleaned)} samples")

    # Balance dataset
    print("Balancing dataset...")
    x_train_balanced, y_train_balanced = balance_dataset(list(x_train_cleaned), list(y_train_filtered))
    x_test_balanced, y_test_balanced = balance_dataset(list(x_test_cleaned), list(y_test_filtered))

    print(f"After balancing:")
    print(f"Train set: {len(x_train_balanced)} samples")
    print(f"Test set: {len(x_test_balanced)} samples")

    # Create train/val/test splits
    print("Creating train/validation/test splits...")
    x_train_final, y_train_final, x_val, y_val, x_test_final, y_test_final = create_splits(
        x_train_balanced + x_test_balanced,
        y_train_balanced + y_test_balanced
    )

    print(f"Final splits:")
    print(f"Train: {len(x_train_final)} samples")
    print(f"Validation: {len(x_val)} samples")
    print(f"Test: {len(x_test_final)} samples")

    # Save processed data
    os.makedirs('processed_data', exist_ok=True)

    # Save as CSV files
    train_df = pd.DataFrame({'text': x_train_final, 'language': y_train_final})
    val_df = pd.DataFrame({'text': x_val, 'language': y_val})
    test_df = pd.DataFrame({'text': x_test_final, 'language': y_test_final})

    train_df.to_csv('processed_data/train.csv', index=False)
    val_df.to_csv('processed_data/val.csv', index=False)
    test_df.to_csv('processed_data/test.csv', index=False)

    # Save language mapping
    lang_mapping_df = pd.DataFrame(list(label_to_lang.items()), columns=['label', 'language'])
    lang_mapping_df.to_csv('processed_data/language_mapping.csv', index=False)

    print("Processed data saved to 'processed_data/' directory")

if __name__ == "__main__":
    main()