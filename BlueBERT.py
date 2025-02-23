import os
import pandas as pd
from typing import List, Tuple, Dict, Any
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classe per gestire l'ingestion dei dati
class DataIngestion:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: pd.DataFrame = None
    
    def load_data(self) -> pd.DataFrame:
        """Carica i dati da un file CSV e ritorna un DataFrame"""
        try:
            self.data = pd.read_csv(self.filepath, header=None, names=["filename", "token", "bio_tag"])
            logger.info(f"Loaded {len(self.data)} rows from {self.filepath}")
        except Exception as e:
            logger.error(f"Error loading data from {self.filepath}: {e}")
            raise
        return self.data

    def preprocess(self) -> pd.DataFrame:
        """Preprocessamento base dei dati (strip e pulizia)"""
        if self.data is None:
            logger.error("Data not loaded. Please load the data first.")
            raise ValueError("Data not loaded. Please load the data first.")
        
        self.data["token"] = self.data["token"].str.strip()
        self.data["bio_tag"] = self.data["bio_tag"].str.strip()
        logger.info("Data preprocessing completed.")
        return self.data


# Classe per il preprocessamento dei dati per BlueBERT
class BioNERPreprocessing:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_data(self, tokens: List[List[str]]) -> Dict[str, Any]:
        """Tokenizza i dati in formato utilizzabile da BlueBERT"""
        encodings = self.tokenizer(
            tokens,  # Assicurati che tokens sia una lista di liste
            truncation=True,
            padding=True,
            is_split_into_words=True,
            max_length=self.max_length
        )
        return encodings


    def create_labels(self, bio_tags: List[str], label2id: Dict[str, int]) -> List[int]:
        """Crea le etichette numeriche per il training"""
        return [label2id.get(tag, -100) for tag in bio_tags]


# Classe per il modello BlueBERT
class BlueBERTModel:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
    
    def initialize_model(self) -> None:
        """Inizializza il modello BlueBERT"""
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Model and Tokenizer initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {e}")
            raise
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Esegue il training del modello"""
        try:
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
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self, test_dataset: Dataset) -> None:
        """Valuta il modello sui dati di test"""
        try:
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer
            )
            results = trainer.evaluate(test_dataset)
            logger.info(f"Test Results: {results}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def compute_metrics(self, p: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Calcola le metriche di precision, recall e F1"""
        predictions, labels = p
        predictions = predictions.argmax(axis=-1)
        
        # Rimuovi le etichette di padding
        true_labels = labels[labels != -100]
        pred_labels = predictions[labels != -100]
        
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# Classe che coordina tutta la pipeline
class BlueBERTPipeline:
    def __init__(self, data_filepath: str):
        self.data_filepath = data_filepath
        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
    
    def run_pipeline(self) -> None:
        """Esegue tutta la pipeline: ingestion, preprocessing, training, evaluation"""
        try:
            # Step 1: Caricamento e preprocessamento dei dati
            ingestion = DataIngestion(self.data_filepath)
            self.data = ingestion.load_data()
            self.data = ingestion.preprocess()

            # Step 2: Preprocessing per BlueBERT
            label2id = {tag: idx for idx, tag in enumerate(self.data["bio_tag"].unique())}
            preprocessing = BioNERPreprocessing(AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT"))
            
            encodings = preprocessing.encode_data(self.data["token"].tolist())
            labels = preprocessing.create_labels(self.data["bio_tag"].tolist(), label2id)
            
            # Step 3: Creazione dei dataset
            dataset = Dataset.from_dict({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels
            })
            
            # Split dei dati in train, validation, test
            self.train_dataset, self.test_dataset = train_test_split(dataset, test_size=0.2)
            self.train_dataset, self.val_dataset = train_test_split(self.train_dataset, test_size=0.2)

            # Step 4: Inizializzazione e training del modello
            self.model = BlueBERTModel()
            self.model.initialize_model()
            self.model.train(self.train_dataset, self.val_dataset)

            # Step 5: Valutazione
            self.model.evaluate(self.test_dataset)
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")


# Esecuzione della pipeline
if __name__== "__main__":
    try:
        pipeline = BlueBERTPipeline(r'C:\Users\batti\bioner\src\token_level_bio.csv')
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")