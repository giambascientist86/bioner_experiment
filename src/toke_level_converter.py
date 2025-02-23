import pandas as pd
import spacy
from typing import List, Tuple
import os

def find_sublist(lst: List[str], sublst: List[str]) -> int:
    """
    Cerca la prima occorrenza di sublst in lst e restituisce l'indice di partenza.
    Se non viene trovato, restituisce -1.
    """
    for i in range(len(lst) - len(sublst) + 1):
        if lst[i:i+len(sublst)] == sublst:
            return i
    return -1

class TokenLevelConverter:
    """
    Classe per convertire un dataset a livello di documento (con colonne di annotazioni)
    in un formato token-level con etichette BIO.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Inizializza il converter caricando il modello spaCy.
        Assicurati di aver installato il modello con: python -m spacy download en_core_web_sm
        """
        self.nlp = spacy.load(model_name)

    def tokenize_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenizza il testo e restituisce una lista di token originali e una versione lower-case per il matching.
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        tokens_lower = [token.text.lower() for token in doc]
        return tokens, tokens_lower

    def tokenize_entity(self, entity_text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenizza il testo di un'entità (annotation) e restituisce i token originali e lower-case.
        """
        doc = self.nlp(entity_text)
        tokens = [token.text for token in doc]
        tokens_lower = [token.text.lower() for token in doc]
        return tokens, tokens_lower

    def convert_row(self, row: pd.Series) -> List[Tuple[str, str]]:
        """
        Converte una singola riga del dataset in una lista di tuple (token, BIO label).
        Usa la colonna 'processed_text' per il testo e le colonne 'Age', 'Sex', 'Symptoms', 'Disease'
        per le annotazioni.
        """
        # Tokenizza il testo preprocessato
        tokens, tokens_lower = self.tokenize_text(row['processed_text'])
        labels = ["O"] * len(tokens)

        # Processa ciascuna tipologia di entità
        for entity_type in ["Age", "Sex", "Symptoms", "Disease"]:
            if pd.isna(row[entity_type]) or str(row[entity_type]).strip() == "":
                continue
            # Le annotazioni potrebbero contenere più entità separate da virgola
            entities = [e.strip() for e in str(row[entity_type]).split(",") if e.strip()]
            for entity in entities:
                # Tokenizza l'entità (in lower-case per matching)
                _, entity_tokens_lower = self.tokenize_entity(entity)
                # Cerca l'entità all'interno dei token del testo
                start_index = find_sublist(tokens_lower, entity_tokens_lower)
                if start_index != -1:
                    # Assegna etichette BIO: B- per il primo token, I- per i successivi
                    bio_label = f"B-{entity_type.upper()}"
                    labels[start_index] = bio_label
                    for i in range(1, len(entity_tokens_lower)):
                        if start_index + i < len(labels):
                            labels[start_index + i] = f"I-{entity_type.upper()}"
                else:
                    # Se l'entità non viene trovata, possiamo stampare un warning o passare
                    print(f"Attenzione: entità '{entity}' di tipo '{entity_type}' non trovata nel testo del file {row['filename']}.")
        return list(zip(tokens, labels))

    def convert_dataset(self, input_csv: str, output_csv: str):
        """
        Legge il dataset in input (CSV), converte ogni riga in formato token-level e salva l'output.
        L'output CSV conterrà le colonne: filename, token, bio_tag.
        """
        df = pd.read_csv(input_csv)
        output_rows = []
        for idx, row in df.iterrows():
            filename = row['filename']
            token_label_pairs = self.convert_row(row)
            for token, label in token_label_pairs:
                output_rows.append({"filename": filename, "token": token, "bio_tag": label})
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_csv, index=False)
        print(f"Dataset token-level salvato in: {output_csv}")

if __name__ == "__main__":
    # Assicurati che il percorso sia corretto
    input_csv_path = os.path.join(r"C:\Users\batti\bioner\src", "feature_engineered_dataset.csv")
    output_csv_path = os.path.join(r"C:\Users\batti\bioner\src", "token_level_dataset.csv")
    
    converter = TokenLevelConverter()
    converter.convert_dataset(input_csv=input_csv_path, output_csv=output_csv_path)
