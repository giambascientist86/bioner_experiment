import pandas as pd
import spacy
from typing import List, Tuple
import os
import difflib

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
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def tokenize_text(self, text: str) -> Tuple[List[str], List[str]]:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        tokens_lower = [token.text.lower() for token in doc]
        return tokens, tokens_lower

    def tokenize_entity(self, entity_text: str) -> Tuple[List[str], List[str]]:
        doc = self.nlp(entity_text)
        tokens = [token.text for token in doc]
        tokens_lower = [token.text.lower() for token in doc]
        return tokens, tokens_lower

    def convert_row(self, row: pd.Series) -> List[Tuple[str, str]]:
        tokens, tokens_lower = self.tokenize_text(row['processed_text'])
        labels = ["O"] * len(tokens)
        token_used = [False] * len(tokens)  # Per tracciare i token già etichettati

        for entity_type in ["Age", "Sex", "Symptoms", "Disease"]:
            if pd.isna(row[entity_type]) or str(row[entity_type]).strip() == "":
                continue

            entities = [e.strip() for e in str(row[entity_type]).split(",") if e.strip()]
            for entity in entities:
                _, entity_tokens_lower = self.tokenize_entity(entity)

                start_index = find_sublist(tokens_lower, entity_tokens_lower)
                
                # Se non trova l'entità con un match esatto, prova un fuzzy match
                if start_index == -1:
                    close_matches = difflib.get_close_matches(entity.lower(), tokens_lower, n=1, cutoff=0.8)
                    if close_matches:
                        start_index = tokens_lower.index(close_matches[0])

                if start_index != -1 and not token_used[start_index]:  # Evita doppie assegnazioni
                    bio_label = f"B-{entity_type.upper()}"
                    labels[start_index] = bio_label
                    token_used[start_index] = True

                    for i in range(1, len(entity_tokens_lower)):
                        if start_index + i < len(labels) and not token_used[start_index + i]:
                            labels[start_index + i] = f"I-{entity_type.upper()}"
                            token_used[start_index + i] = True
                else:
                    print(f"⚠️ Warning: '{entity}' ({entity_type}) non trovato nel testo del file {row['filename']}.")

        return list(zip(tokens, labels))

    def convert_dataset(self, input_csv: str, output_csv: str):
        df = pd.read_csv(input_csv)
        output_rows = []
        
        for idx, row in df.iterrows():
            filename = row['filename']
            token_label_pairs = self.convert_row(row)
            
            for token, label in token_label_pairs:
                output_rows.append({"filename": filename, "token": token, "bio_tag": label})
        
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_csv, index=False)
        print(f"✅ Dataset token-level salvato in: {output_csv}")

if __name__ == "__main__":
    input_csv_path = os.path.join(r"C:\Users\batti\bioner\src", "feature_engineered_dataset.csv")
    output_csv_path = os.path.join(r"C:\Users\batti\bioner\src", "token_level_dataset_v2.csv")
    
    converter = TokenLevelConverter()
    converter.convert_dataset(input_csv=input_csv_path, output_csv=output_csv_path)
