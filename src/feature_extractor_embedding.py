import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, df: pd.DataFrame, text_column: str = "processed_text"):
        """
        Initialize FeatureExtractor with a DataFrame containing text data.
        :param df: DataFrame containing the dataset
        :param text_column: Name of the column with preprocessed text
        """
        self.df = df
        self.text_column = text_column
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.biowordvec = None

    def compute_tfidf(self):
        """Compute TF-IDF embeddings for text."""
        print("Extracting TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(self.df[self.text_column]).toarray()
        return pd.DataFrame(tfidf_features, columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])])

    def compute_scibert_embeddings(self):
        """Compute SciBERT embeddings for text."""
        print("Extracting SciBERT embeddings...")
        embeddings = []
        for text in tqdm(self.df[self.text_column]):
            tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**tokens)
            emb = output.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb)
        return pd.DataFrame(embeddings, columns=[f"scibert_{i}" for i in range(embeddings[0].shape[0])])

    def load_biowordvec(self, path: str):
        """Load BioWordVec pre-trained embeddings."""
        print("Loading BioWordVec...")
        self.biowordvec = KeyedVectors.load_word2vec_format(path, binary=True)
    
    def compute_biowordvec_embeddings(self):
        """Compute BioWordVec embeddings for text."""
        if self.biowordvec is None:
            raise ValueError("BioWordVec model not loaded. Use load_biowordvec() first.")
        print("Extracting BioWordVec embeddings...")
        embeddings = []
        for text in tqdm(self.df[self.text_column]):
            words = text.split()
            word_vectors = [self.biowordvec[word] for word in words if word in self.biowordvec]
            if word_vectors:
                emb = np.mean(word_vectors, axis=0)
            else:
                emb = np.zeros(self.biowordvec.vector_size)
            embeddings.append(emb)
        return pd.DataFrame(embeddings, columns=[f"biowordvec_{i}" for i in range(embeddings[0].shape[0])])

    def extract_features(self):
        """Generate all features and return the updated DataFrame."""
        print("Starting Feature Extraction...")
        tfidf_df = self.compute_tfidf()
        scibert_df = self.compute_scibert_embeddings()
        
        if self.biowordvec:
            biowordvec_df = self.compute_biowordvec_embeddings()
            self.df = pd.concat([self.df, tfidf_df, scibert_df, biowordvec_df], axis=1)
        else:
            self.df = pd.concat([self.df, tfidf_df, scibert_df], axis=1)
        
        return self.df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\batti\bioner\src\processed_data.csv")
    fe = FeatureExtractor(df)
    df_features = fe.extract_features()
    df_features.to_csv(r"C:\Users\batti\bioner\src\feature_engineered_dataset.csv", index=False)
    print("Feature extraction completed.")