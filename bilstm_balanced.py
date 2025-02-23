import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report

# -------------------- DataLoader --------------------
class DataLoader:
    def __init__(self, input_csv: str, max_length: int = None, oversample: bool = False):
        self.input_csv = input_csv
        self.max_length = max_length
        self.oversample = oversample
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.X = None
        self.Y = None
        self.max_len = None
        self._load_and_preprocess()
    
    def _load_and_preprocess(self):
        df = pd.read_csv(self.input_csv)
        grouped = df.groupby("filename").agg({
            "token": lambda x: " ".join(x),
            "bio_tag": lambda x: " ".join(x)
        }).reset_index()
        
        self.tokens = grouped["token"].apply(lambda x: x.split()).tolist()
        self.labels = grouped["bio_tag"].apply(lambda x: x.split()).tolist()
        
        unique_words = set(word for sent in self.tokens for word in sent)
        self.word2idx = {w: i+1 for i, w in enumerate(unique_words)}
        unique_tags = set(tag for sent in self.labels for tag in sent)
        self.tag2idx = {t: i for i, t in enumerate(unique_tags)}
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
        
        self.max_len = self.max_length if self.max_length else max(len(sent) for sent in self.tokens)
        
        self.X = [[self.word2idx[w] for w in sent] for sent in self.tokens]
        self.X = pad_sequences(self.X, maxlen=self.max_len, padding="post")
        
        self.Y = [[self.tag2idx[t] for t in sent] for sent in self.labels]
        self.Y = pad_sequences(self.Y, maxlen=self.max_len, padding="post")
        self.Y = [to_categorical(seq, num_classes=len(self.tag2idx)) for seq in self.Y]
        
        if self.oversample:
            self._oversample_minority_classes()
    
    def _oversample_minority_classes(self):
        label_counts = Counter(tag for sent in self.labels for tag in sent)
        max_count = max(label_counts.values())
        
        new_X, new_Y = [], []
        for x, y in zip(self.X, self.Y):
            label_distribution = Counter(np.argmax(y, axis=-1))
            oversample_factor = max_count / max(label_distribution.values())
            if oversample_factor > 1:
                for _ in range(int(oversample_factor)):
                    new_X.append(x)
                    new_Y.append(y)
            else:
                new_X.append(x)
                new_Y.append(y)
        
        self.X = np.array(new_X)
        self.Y = np.array(new_Y)
    
    def get_data(self):
        return self.X, self.Y
    
    def train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state)
        return X_train, X_test, Y_train, Y_test
    
    def save_mappings(self, folder="models"):
        os.makedirs(folder, exist_ok=True)
        pickle.dump(self.word2idx, open(os.path.join(folder, "word2idx.pkl"), "wb"))
        pickle.dump(self.tag2idx, open(os.path.join(folder, "tag2idx.pkl"), "wb"))
        pickle.dump(self.idx2tag, open(os.path.join(folder, "idx2tag.pkl"), "wb"))
        print("Mappings salvate in", folder)

# -------------------- ModelBuilder --------------------
class ModelBuilder:
    def __init__(self, input_dim, output_dim, max_len, n_tags):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.n_tags = n_tags
    
    def build_model(self):
        model = Sequential([
            Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                      input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            TimeDistributed(Dense(self.n_tags, activation="sigmoid"))
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        return model

# -------------------- Trainer --------------------
class Trainer:
    def __init__(self, model, X_train, Y_train, X_val, Y_val, batch_size=32, epochs=10):
        self.model = model
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        self.X_val = np.array(X_val)
        self.Y_val = np.array(Y_val)
        self.batch_size = batch_size
        self.epochs = epochs
    
    def train(self):
        history = self.model.fit(self.X_train, self.Y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(self.X_val, self.Y_val),
                                 verbose=1)
        return history

# -------------------- Evaluator --------------------
class Evaluator:
    def __init__(self, model, X_test, Y_test, idx2tag):
        self.model = model
        self.X_test = np.array(X_test)
        self.Y_test = np.array(Y_test)
        self.idx2tag = idx2tag
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(self.Y_test, axis=-1)
        
        y_pred_labels = [[self.idx2tag[idx] for idx in seq] for seq in y_pred]
        y_true_labels = [[self.idx2tag[idx] for idx in seq] for seq in y_true]
        
        report = classification_report(y_true_labels, y_pred_labels)
        print("Classification Report:\n", report)
        return report

# -------------------- Pipeline --------------------
class BioNERPipeline:
    def __init__(self, input_csv: str, oversample: bool = True):
        self.input_csv = input_csv
        self.oversample = oversample
    
    def run(self):
        print("Caricamento e preprocessing dei dati...")
        data_loader = DataLoader(self.input_csv, oversample=self.oversample)
        X, Y = data_loader.get_data()
        X_train, X_test, Y_train, Y_test = data_loader.train_test_split()
        data_loader.save_mappings()
        
        print("Costruzione del modello...")
        model_builder = ModelBuilder(len(data_loader.word2idx) + 1, 64, data_loader.max_len, len(data_loader.tag2idx))
        model = model_builder.build_model()
        
        print("Training del modello...")
        trainer = Trainer(model, X_train, Y_train, X_test, Y_test, epochs=10)
        trainer.train()
        model.save("models/bilstm_crf_ner.h5")
        
        print("Valutazione del modello...")
        evaluator = Evaluator(model, X_test, Y_test, data_loader.idx2tag)
        evaluator.evaluate()

if __name__ == "__main__":
    BioNERPipeline(r"C:\Users\batti\bioner\src\token_level_dataset_v2.csv").run()
