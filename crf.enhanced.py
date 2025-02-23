import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
            self.data = pd.read_csv(self.file_path, encoding="utf-8").dropna()
        except UnicodeDecodeError:
            self.data = pd.read_csv(self.file_path, encoding="ISO-8859-1").dropna()
        
        if "filename" not in self.data.columns or "token" not in self.data.columns or "bio_tag" not in self.data.columns:
            raise ValueError("Il dataset deve contenere le colonne: 'filename', 'token', 'bio_tag'")
        
        self.data.fillna("O", inplace=True)  # Riempi eventuali NaN con "O"
        return self.data

    def get_sentences(self) -> List[List[Tuple[str, str]]]:
        """Raggruppa i dati per filename e restituisce frasi con token e BIO-tag."""
        if self.data is None:
            raise ValueError("Caricare i dati prima di chiamare get_sentences()")
        
        sentences = (
            self.data.groupby("filename")
            .apply(lambda x: list(zip(x["token"], x["bio_tag"])))
            .tolist()
        )
        return sentences


class FeatureExtractor:
    """Estrae feature dai token per il modello CRF."""

    @staticmethod
    def extract_features(sentence: List[Tuple[str, str]], i: int) -> dict:
        """Estrae feature per il token in posizione i."""
        word = sentence[i][0]
        features = {
            "word.lower()": word.lower(),
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "BOS": i == 0,
            "EOS": i == len(sentence) - 1,
        }
        # Aggiunta di feature contestuali
        if i > 0:
            prev_word = sentence[i - 1][0]
            features["-1:word.lower()"] = prev_word.lower()
        else:
            features["BOS"] = True

        if i < len(sentence) - 1:
            next_word = sentence[i + 1][0]
            features["+1:word.lower()"] = next_word.lower()
        else:
            features["EOS"] = True

        return features

    def transform(self, sentences: List[List[Tuple[str, str]]]) -> Tuple[List[List[dict]], List[List[str]]]:
        """Trasforma le frasi in feature e label per il modello."""
        X = [[self.extract_features(sentence, i) for i in range(len(sentence))] for sentence in sentences]
        y = [[label for _, label in sentence] for sentence in sentences]
        return X, y


class CRFModel:
    """Modello CRF per Named Entity Recognition."""

    def __init__(self):
        self.model = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100)

    def train(self, X_train: List[List[dict]], y_train: List[List[str]]):
        """Addestra il modello CRF."""
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, "crf_model.pkl")
        print("✅ Modello CRF salvato come 'crf_model.pkl'")

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
        
        print("🔄 Estrazione feature...")
        sentences = self.dataset_loader.get_sentences()
        X, y = self.feature_extractor.transform(sentences)
        
        print("✂️ Suddivisione train/test preservando filename...")
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        groups = [sentences[i][0][0] for i in range(len(sentences))]
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]
        
        print("🚀 Addestramento modello...")
        self.model.train(X_train, y_train)
        
        print("🧠 Effettuazione predizioni...")
        y_pred = self.model.predict(X_test)
        
        print("📊 Valutazione...")
        self.evaluator.evaluate(y_test, y_pred)


# ✅ Avvio della pipeline
if __name__ == "__main__":
    pipeline = Pipeline(r"C:\Users\batti\bioner\src\token_level_dataset.csv")
    pipeline.run()
