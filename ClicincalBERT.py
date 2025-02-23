import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import logging
from typing import List, Tuple
import pymongo
import os

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connessione MongoDB per MLOps
class MLOpsDB:
    def __init__(self, db_url: str = "mongodb://localhost:27017/", db_name: str = "model_db"):
        self.client = pymongo.MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db["models"]

    def save_model(self, model_name: str, model: BertForTokenClassification, model_version: str, metrics: dict):
        """Salva il modello nel database MLOps con metriche"""
        model_data = {
            "model_name": model_name,
            "model_version": model_version,
            "metrics": metrics
        }
        self.collection.insert_one(model_data)
        logger.info(f"Modello {model_name} versione {model_version} salvato nel database.")

    def load_model(self, model_name: str, model_version: str) -> BertForTokenClassification:
        """Carica il modello dal database MLOps"""
        model_data = self.collection.find_one({"model_name": model_name, "model_version": model_version})
        if model_data:
            logger.info(f"Modello {model_name} versione {model_version} caricato dal database.")
            return model_data['model']
        else:
            logger.warning(f"Modello {model_name} versione {model_version} non trovato nel database.")
            return None


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Carica i dati dal file CSV"""
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"Dati caricati da {self.file_path}")
            return data
        except Exception as e:
            logger.error(f"Errore nel caricamento dei dati: {e}")
            raise

    def preprocess(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Pre-processa i dati per il modello"""
        logger.info("Preprocessing dei dati completato.")
        X = data['text'].tolist()  # Testo
        y = data['labels'].tolist()  # Etichette
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


class Preprocessor:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def preprocess_text(self, texts: List[str]) -> List[dict]:
        """Tokenizza il testo"""
        try:
            encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            logger.info(f"Tokenizzazione completata per {len(texts)} testi.")
            return encoded_texts
        except Exception as e:
            logger.error(f"Errore nella tokenizzazione del testo: {e}")
            raise


class ClinicalBERTModel:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.model = BertForTokenClassification.from_pretrained(model_name)

    def get_model(self):
        """Restituisce il modello caricato"""
        return self.model


class TrainerClass:
    def __init__(self, model, train_data, eval_data):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data

    def train(self, epochs: int = 3, learning_rate: float = 5e-5):
        """Configura e allena il modello"""
        try:
            training_args = TrainingArguments(
                output_dir="./results", 
                num_train_epochs=epochs, 
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                evaluation_strategy="epoch",
                learning_rate=learning_rate
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_data,
                eval_dataset=self.eval_data
            )
            trainer.train()
            logger.info("Modello addestrato con successo.")
            return trainer
        except Exception as e:
            logger.error(f"Errore nell'addestramento: {e}")
            raise


class Evaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def evaluate(self):
        """Esegue la valutazione del modello"""
        try:
            predictions = self.model.predict(self.test_data)
            report = classification_report(self.test_data.labels, predictions)
            logger.info(f"Classificazione completata:\n{report}")
            return report
        except Exception as e:
            logger.error(f"Errore nella valutazione: {e}")
            raise


class HyperparameterTuning:
    def __init__(self, model, param_grid: dict):
        self.model = model
        self.param_grid = param_grid

    def grid_search(self, train_data, eval_data):
        """Esegue la ricerca degli iperparametri tramite GridSearchCV"""
        try:
            grid_search = GridSearchCV(self.model, self.param_grid, cv=3)
            grid_search.fit(train_data, eval_data)
            logger.info(f"GridSearch completato con i migliori parametri: {grid_search.best_params_}")
            return grid_search.best_params_
        except Exception as e:
            logger.error(f"Errore nel GridSearchCV: {e}")
            raise


# Main function to run the entire pipeline
def main(file_path: str):
    try:
        # Caricamento dei dati
        data_loader = DataLoader(file_path)
        data = data_loader.load_data()
        X_train, X_test, y_train, y_test = data_loader.preprocess(data)

        # Preprocessing del testo
        preprocessor = Preprocessor()
        train_data = preprocessor.preprocess_text(X_train)
        test_data = preprocessor.preprocess_text(X_test)

        # Caricamento del modello
        model = ClinicalBERTModel().get_model()

        # Hyperparameter Tuning
        param_grid = {
            'learning_rate': [5e-5, 3e-5],
            'num_train_epochs': [3, 5]
        }
        tuner = HyperparameterTuning(model, param_grid)
        best_params = tuner.grid_search(train_data, test_data)

        # Allenamento del modello con i migliori parametri
        trainer = TrainerClass(model, train_data, test_data)
        trainer.train(epochs=best_params['num_train_epochs'], learning_rate=best_params['learning_rate'])

        # Salvataggio modello con MLOps
        mlops_db = MLOpsDB()
        metrics = {"accuracy": 0.9}  # Esempio di metrica, da sostituire con metrica reale
        mlops_db.save_model(model_name="ClinicalBERT", model=model, model_version="1.0", metrics=metrics)

        # Valutazione del modello
        evaluator = Evaluator(model, test_data)
        evaluation_report = evaluator.evaluate()
        
        # Stampa report
        print(evaluation_report)

    except Exception as e:
        logger.error(f"Errore nell'esecuzione della pipeline: {e}")


# Esegui il main
if __name__ == "__main__":
    main(r"C:\Users\batti\bioner\models\clinical_BERT\processed_dataset_clinicalbert.csv")
