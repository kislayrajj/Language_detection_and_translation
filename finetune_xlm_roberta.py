import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

class LanguageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_data(data_dir='processed_data'):
    """Load processed data"""
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # Create label mappings
    all_languages = sorted(train_df['language'].unique())
    label_to_id = {lang: i for i, lang in enumerate(all_languages)}
    id_to_label = {i: lang for lang, i in label_to_id.items()}

    # Convert labels to IDs
    train_df['label'] = train_df['language'].map(label_to_id)
    val_df['label'] = val_df['language'].map(label_to_id)
    test_df['label'] = test_df['language'].map(label_to_id)

    print(f"Number of languages: {len(all_languages)}")
    print(f"Languages: {all_languages[:10]}...")

    return train_df, val_df, test_df, label_to_id, id_to_label

def main():
    # Load data
    print("Loading processed data...")
    train_df, val_df, test_df, label_to_id, id_to_label = load_data()

    # Initialize tokenizer and model
    print("Loading XLM-RoBERTa model...")
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id)
    )

    # Create datasets
    train_dataset = LanguageDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )

    val_dataset = LanguageDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )

    test_dataset = LanguageDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Reduced epochs for disk space
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=8,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",  # Updated parameter name
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,  # Save fewer checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=3e-5,  # Slightly higher learning rate
        fp16=False,  # Disable mixed precision for compatibility
        dataloader_num_workers=0,  # Windows compatibility
        gradient_accumulation_steps=2,  # Effective batch size 16
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # Save the model
    print("Saving model...")
    model.save_pretrained('./xlm_roberta_language_model')
    tokenizer.save_pretrained('./xlm_roberta_language_model')

    # Save label mappings
    import json
    with open('./xlm_roberta_language_model/label_to_id.json', 'w') as f:
        json.dump(label_to_id, f)

    with open('./xlm_roberta_language_model/id_to_label.json', 'w') as f:
        json.dump(id_to_label, f)

    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()