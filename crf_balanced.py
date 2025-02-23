import pandas as pd
import os
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from typing import List, Tuple


class DatasetLoader:
    """Carica e pre-processa il dataset BIO-tagged."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Carica il dataset da CSV e lo pre-processa."""
        try:
            self.data = pd.read_csv(self.file_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.data = pd.read_csv(self.file_path, encoding="ISO-8859-1")
        
        if "token" not in self.data.columns or "bio_tag" not in self.data.columns:
            raise ValueError("Il dataset non contiene le colonne richieste: 'token', 'bio_tag'")
        
        self.data.fillna("O", inplace=True)  # Riempi eventuali NaN con "O"
        return self.data

    def get_sentences(self) -> List[List[Tuple[str, str]]]:
        """Raggruppa i dati per filename e restituisce una lista di frasi con token e BIO-tag."""
        if self.data is None:
            raise ValueError("Caricare i dati prima di chiamare get_sentences()")
        
        sentences = (
            self.data.groupby("filename")
            .apply(lambda x: list(zip(x["token"], x["bio_tag"])))
            .tolist()
        )
        return sentences

    def get_class_distribution(self) -> Counter:
        """Restituisce la distribuzione delle classi BIO."""
        return Counter(self.data["bio_tag"].values)


class FeatureExtractor:
    """Estrae feature dai token per il modello CRF."""

    @staticmethod
    def extract_features(sentence: List[Tuple[str, str]], i: int) -> dict:
        """Estrae feature per il token in posizione i in una frase."""
        word = sentence[i][0]
        return {
            "word.lower()": word.lower(),
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "BOS": i == 0,  # Inizio della frase
            "EOS": i == len(sentence) - 1  # Fine della frase
        }

    def transform(self, sentences: List[List[Tuple[str, str]]]) -> Tuple[List[List[dict]], List[List[str]]]:
        """Trasforma le frasi in feature e label per il modello."""
        X = [[self.extract_features(sentence, i) for i in range(len(sentence))] for sentence in sentences]
        y = [[label for _, label in sentence] for sentence in sentences]
        return X, y


class DataBalancer:
    """Bilancia le classi rare con oversampling."""

    @staticmethod
    def balance_data(sentences: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        """Aumenta la presenza delle classi minoritarie duplicando frasi con tag rari."""
        
        # Conta la frequenza dei BIO-tag
        tag_counts = Counter(tag for sentence in sentences for _, tag in sentence)
        rare_tags = {tag for tag, count in tag_counts.items() if count < 100}  # Soglia arbitraria

        # Identifica frasi con almeno un tag raro
        rare_sentences = [s for s in sentences if any(tag in rare_tags for _, tag in s)]
        
        # Aggiunge copie delle frasi con tag rari fino a eguagliare le classi maggioritarie
        oversampled_sentences = sentences + rare_sentences * 2  # Raddoppia le frasi con tag rari
        np.random.shuffle(oversampled_sentences)  # Mescola il dataset
        return oversampled_sentences


class CRFModel:
    """Modello CRF per Named Entity Recognition."""

    def __init__(self):
        self.model = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100)

    def train(self, X_train: List[List[dict]], y_train: List[List[str]]):
        """Addestra il modello CRF."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test: List[List[dict]]) -> List[List[str]]:
        """Effettua previsioni su nuovi dati."""
        return self.model.predict(X_test)


class Evaluator:
    """Calcola metriche di valutazione per il modello CRF."""

    @staticmethod
    def evaluate(y_true: List[List[str]], y_pred: List[List[str]]):
        """Stampa il report di classificazione."""
        print(flat_classification_report(y_true, y_pred))


class Pipeline:
    """Pipeline completa per il training e la valutazione del modello CRF."""

    def __init__(self, file_path: str):
        self.dataset_loader = DatasetLoader(file_path)
        self.feature_extractor = FeatureExtractor()
        self.model = CRFModel()
        self.evaluator = Evaluator()

    def run(self):
        """Esegue la pipeline end-to-end."""
        print("📂 Caricamento dataset...")
        self.dataset_loader.load_data()
        print("📊 Distribuzione classi prima del bilanciamento:", self.dataset_loader.get_class_distribution())

        print("🔄 Estrazione feature...")
        sentences = self.dataset_loader.get_sentences()

        print("⚖️ Bilanciamento classi...")
        balanced_sentences = DataBalancer.balance_data(sentences)

        print("📊 Distribuzione classi dopo il bilanciamento:", Counter(tag for s in balanced_sentences for _, tag in s))

        X, y = self.feature_extractor.transform(balanced_sentences)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("🚀 Addestramento modello...")
        self.model.train(X_train, y_train)

        print("🧠 Effettuazione predizioni...")
        y_pred = self.model.predict(X_test)

        print("📊 Valutazione...")
        self.evaluator.evaluate(y_test, y_pred)


if __name__ == "__main__":
    pipeline = Pipeline(r"C:\Users\batti\bioner\src\token_level_dataset_v2.csv")
    pipeline.run()
