import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Classe per il dataset personalizzato
class ClinicalBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenizzazione del testo
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Creazione delle etichette corrispondenti
        labels = torch.tensor(self.labels[idx])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

# Classe per il modello ClinicalBERT
class ClinicalBERTModel:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=5)

    def train(self, train_dataset, val_dataset, output_dir="./results", epochs=3, batch_size=16):
        # Parametri di addestramento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_dir='./logs',
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

    def evaluate(self, eval_dataset):
        trainer = Trainer(model=self.model)
        result = trainer.evaluate(eval_dataset)
        return result

# Funzione di pre-processing per le etichette
def convert_labels_to_ids(labels, label_map):
    return [[label_map[label] for label in row] for row in labels]

# Caricamento del dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['text'].tolist(), df['labels'].apply(eval).tolist()

def main():
    # Caricamento e pre-processing del dataset
    texts, labels = load_data(r"C:\Users\batti\bioner\models\clinical_BERT\processed_dataset_clinicalbert.csv")
    
    # Mappatura delle etichette (esempio)
    label_map = label_map = {'O': 0, 'B-AGE': 1, 'I-AGE': 2, 'B-SYMPTOMS': 3, 'I-SYMPTOMS': 4, 'B-DISEASE': 5, 'I-DISEASE': 6,'B-SEX':7, 'I-SEX':8}
    labels = convert_labels_to_ids(labels, label_map)

    # Split del dataset in training e validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)
    
    # Creazione dei dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = ClinicalBERTDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClinicalBERTDataset(val_texts, val_labels, tokenizer)
    
    # Inizializzazione e addestramento del modello
    model = ClinicalBERTModel()
    model.train(train_dataset, val_dataset)

    # Valutazione del modello
    evaluation_results = model.evaluate(val_dataset)
    print(evaluation_results)

if __name__ == "__main__":
    main()
