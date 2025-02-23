import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Carica i dati da un file in formato CoNLL-like.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Carica il dataset da un file CSV in formato CoNLL-like.
        """
        # Carica il file in formato CoNLL-like con colonne 'Token' e 'BIO_Tag'
        df = pd.read_csv(self.file_path, sep=" ", header=None, names=["Token", "BIO_Tag"])
        df = df.dropna()  # Rimuovi eventuali righe vuote
        return df


class Tokenizer:
    """
    Tokenizza i dati per il modello ClinicalBERT utilizzando WordPiece.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, df: pd.DataFrame) -> List[dict]:
        """
        Tokenizza i dati (Token, BIO_Tag) in un formato compatibile con BERT.
        Restituisce una lista di dizionari con gli ID dei token e le etichette BIO.
        """
        sentences = df.groupby("Filename")
        tokenized_inputs = []

        for _, group in sentences:
            tokens = group["Token"].tolist()
            labels = group["BIO_Tag"].tolist()

            # Tokenizzazione con WordPiece
            encoding = self.tokenizer(tokens, truncation=True, padding="max_length", is_split_into_words=True)
            tokenized_inputs.append((encoding, labels))

        return tokenized_inputs


class DatasetPreparation:
    """
    Prepara il dataset per l'addestramento di ClinicalBERT.
    """
    def __init__(self, tokenizer: Tokenizer, data_loader: DataLoader):
        self.tokenizer = tokenizer
        self.data_loader = data_loader

    def prepare_dataset(self) -> DatasetDict:
        """
        Prepara il dataset per l'addestramento e la valutazione.
        """
        # Carica e preprocessa il dataset
        df = self.data_loader.load_data()

        # Tokenizza il dataset
        tokenized_data = self.tokenizer.tokenize(df)

        # Splitting in train e test
        train_data, test_data = train_test_split(tokenized_data, test_size=0.1)

        # Converti in Hugging Face Dataset
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        return dataset


class ClinicalBERTModel:
    """
    Carica e configura il modello ClinicalBERT.
    """
    def __init__(self, model_name: str):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification for BIO

    def configure_training(self, train_dataset: Dataset, output_dir: str) -> Trainer:
        """
        Configura il training e restituisce un Trainer per l'addestramento del modello.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_steps=1000
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            tokenizer=self.model.config.tokenizer_class
        )

        return trainer


class ModelTrainer:
    """
    Gestisce il processo di training del modello ClinicalBERT.
    """
    def __init__(self, model: ClinicalBERTModel, dataset: DatasetDict, output_dir: str):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir

    def train(self):
        """
        Esegue l'addestramento del modello.
        """
        trainer = self.model.configure_training(self.dataset["train"], self.output_dir)
        trainer.train()


class ModelSaver:
    """
    Salva il modello allenato su disco.
    """
    def __init__(self, model: ClinicalBERTModel, output_dir: str):
        self.model = model
        self.output_dir = output_dir

    def save_model(self):
        """
        Salva il modello allenato.
        """
        self.model.model.save_pretrained(self.output_dir)


class Pipeline:
    """
    Pipeline end-to-end per il caricamento, preprocessing, training e salvataggio del modello.
    """
    def __init__(self, input_path: str, model_name: str, output_dir: str):
        self.input_path = input_path
        self.model_name = model_name
        self.output_dir = output_dir

    def run(self):
        # Step 1: Carica e preprocessa il dataset
        data_loader = DataLoader(self.input_path)
        tokenizer = Tokenizer(self.model_name)
        dataset_preparation = DatasetPreparation(tokenizer, data_loader)
        dataset = dataset_preparation.prepare_dataset()

        # Step 2: Crea e configura il modello
        model = ClinicalBERTModel(self.model_name)

        # Step 3: Allena il modello
        trainer = ModelTrainer(model, dataset, self.output_dir)
        trainer.train()

        # Step 4: Salva il modello allenato
        model_saver = ModelSaver(model, self.output_dir)
        model_saver.save_model()


# Esecuzione della pipeline
input_path = r"C:\Users\batti\bioner\models\clinical_BERT\ner_dataset.conll"  # Sostituisci con il tuo percorso del file CoNLL-like
model_name = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT da Hugging Face
output_dir = "output/clinicalbert_model"

pipeline = Pipeline(input_path, model_name, output_dir)
pipeline.run()
