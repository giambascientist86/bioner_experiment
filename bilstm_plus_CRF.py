import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# -------------------- DataLoader --------------------
class DataLoaderPyTorch:
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
        self.X = self._pad_sequences(self.X, self.max_len)
        
        self.Y = [[self.tag2idx[t] for t in sent] for sent in self.labels]
        self.Y = self._pad_sequences(self.Y, self.max_len)
        
        if self.oversample:
            self._oversample_minority_classes()
    
    def _pad_sequences(self, sequences, maxlen):
        return [seq + [0] * (maxlen - len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in sequences]
    
    def _oversample_minority_classes(self):
        label_counts = Counter(tag for sent in self.labels for tag in sent)
        max_count = max(label_counts.values())
        
        new_X, new_Y = [], []
        for x, y in zip(self.X, self.Y):
            label_distribution = Counter(y)
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

# -------------------- BiLSTM + CRF Model --------------------
class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_tags, dropout=0.1):
        super(BiLSTMCRF, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, output_dim)
        self.lstm = nn.LSTM(output_dim, hidden_dim, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_tags)  # BiLSTM output is concatenated hidden states
        self.crf = CRF(n_tags, batch_first=True)
    
    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.fc(lstm_out)
        return emissions
    
    def loss(self, x, y):
        emissions = self.forward(x)
        return -self.crf(emissions, y)
    
    def decode(self, x):
        emissions = self.forward(x)
        return self.crf.decode(emissions)

# -------------------- Trainer --------------------
class Trainer:
    def __init__(self, model, X_train, Y_train, X_val, Y_val, batch_size=32, epochs=10, lr=0.001):
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.long)
        self.Y_train = torch.tensor(Y_train, dtype=torch.long)
        self.X_val = torch.tensor(X_val, dtype=torch.long)
        self.Y_val = torch.tensor(Y_val, dtype=torch.long)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(self.X_train), self.batch_size):
                X_batch = self.X_train[i:i + self.batch_size].to(self.device)
                Y_batch = self.Y_train[i:i + self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                loss = self.model.loss(X_batch, Y_batch)
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.4f}")
        
        torch.save(self.model.state_dict(), "models/bilstm_crf_ner.pth")
        print("Modello salvato!")

# -------------------- Evaluator --------------------
class Evaluator:
    def __init__(self, model, X_test, Y_test, idx2tag):
        self.model = model
        self.X_test = torch.tensor(X_test, dtype=torch.long)
        self.Y_test = torch.tensor(Y_test, dtype=torch.long)
        self.idx2tag = idx2tag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def evaluate(self):
        self.model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for i in range(0, len(self.X_test), 32):
                X_batch = self.X_test[i:i + 32].to(self.device)
                Y_batch = self.Y_test[i:i + 32].to(self.device)
                
                predictions = self.model.decode(X_batch)
                y_pred.extend(predictions)
                y_true.extend(Y_batch.cpu().numpy())
        
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
        data_loader = DataLoaderPyTorch(self.input_csv, oversample=self.oversample)
        X, Y = data_loader.get_data()
        X_train, X_test, Y_train, Y_test = data_loader.train_test_split()
        data_loader.save_mappings()
        
        print("Costruzione del modello...")
        model = BiLSTMCRF(input_dim=len(data_loader.word2idx) + 1,
                          output_dim=64, hidden_dim=64,
                          n_tags=len(data_loader.tag2idx))
        
        print("Training del modello...")
        trainer = Trainer(model, X_train, Y_train, X_test, Y_test, epochs=10)
        trainer.train()
        
        print("Valutazione del modello...")
        evaluator = Evaluator(model, X_test, Y_test, data_loader.idx2tag)
        evaluator.evaluate()

if __name__ == "__main__":
    BioNERPipeline(r"C:\Users\batti\bioner\src\token_level_dataset_v2.csv").run()
