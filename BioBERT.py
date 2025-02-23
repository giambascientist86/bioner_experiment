import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support
import os

# Verifica se Accelerate è installato correttamente
try:
    import accelerate
    assert accelerate.__version__ >= "0.26.0"
except (ImportError, AssertionError):
    raise ImportError("Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`. Please run `pip install 'accelerate>=0.26.0'`.")

# Classe per gestire l'ingestion dei dati
class DataIngestion:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
    
    def load_data(self):
        # Caricamento del dataset
        self.data = pd.read_csv(self.filepath, header=None, names=["filename", "token", "bio_tag"])
        print(f"Loaded {len(self.data)} rows")
        return self.data

    def preprocess(self):
        # Trasformazioni di preprocessing (se necessarie)
        self.data["token"] = self.data["token"].str.strip()
        self.data["bio_tag"] = self.data["bio_tag"].str.strip()
        return self.data


# Classe per il preprocessamento dei dati per il BioBERT
class BioNERPreprocessing:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def encode_data(self, data):
        tokenized_texts = data['token'].fillna("").apply(lambda x: x.split()).tolist()
        encodings = self.tokenizer(
            tokenized_texts,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encodings

    def create_labels(self, data, label2id):
        labels = [label2id.get(tag, -100) for tag in data['bio_tag'].tolist()]
        return labels


# Classe per definire e allenare il modello BioBERT
class BioBERTModel:
    def __init__(self, model_name="dmis-lab/biobert-v1.1", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
    
    def initialize_model(self):
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Model and Tokenizer initialized")
    
    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir='./results',          
            num_train_epochs=3,              
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=16,
            warmup_steps=500,                
            weight_decay=0.01,              
            logging_dir='./logs',            
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )
        
        trainer = Trainer(
            model=self.model,                 
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=val_dataset,          
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()

    def evaluate(self, test_dataset):
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer
        )
        results = trainer.evaluate(test_dataset)
        print(f"Test Results: {results}")
    
    def compute_metrics(self, p):
        predictions, labels = p.predictions, p.label_ids

        # Controllo delle dimensioni
        print(f"Predictions shape: {predictions.shape}")  # (batch_size, seq_length, num_labels)
        print(f"Labels shape: {labels.shape}")  # Dovrebbe essere (batch_size, seq_length)

        predictions = predictions.argmax(axis=-1)

        true_labels = labels[labels != -100]
        pred_labels = predictions[labels != -100]

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# Classe principale che unisce tutti i passi
class BioNERPipeline:
    def __init__(self, data_filepath):
        self.data_filepath = data_filepath
        self.model = None
        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def run_pipeline(self):
        ingestion = DataIngestion(self.data_filepath)
        self.data = ingestion.load_data()
        self.data = ingestion.preprocess()

        label2id = {tag: idx for idx, tag in enumerate(self.data["bio_tag"].unique())}
        preprocessing = BioNERPreprocessing(AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1"))
        
        encodings = preprocessing.encode_data(self.data)
        labels = preprocessing.create_labels(self.data, label2id)
        
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"].tolist(),
            "attention_mask": encodings["attention_mask"].tolist(),
            "labels": labels
        })
        
        # Split del dataset usando il metodo nativo di Hugging Face
        dataset = dataset.train_test_split(test_size=0.2)
        train_val_split = dataset["train"].train_test_split(test_size=0.2)
        
        self.train_dataset = train_val_split["train"]
        self.val_dataset = train_val_split["test"]
        self.test_dataset = dataset["test"]

        self.model = BioBERTModel()
        self.model.initialize_model()
        self.model.train(self.train_dataset, self.val_dataset)
        self.model.evaluate(self.test_dataset)


# Esecuzione della pipeline
if __name__ == "__main__":
    pipeline = BioNERPipeline(r"C:\Users\batti\bioner\src\token_level_bio.csv")
    pipeline.run_pipeline()
