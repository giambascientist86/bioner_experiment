import pandas as pd
import spacy
from spacy.tokens import DocBin
from pathlib import Path
from typing import List, Tuple

class DataProcessor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.df = pd.read_csv(input_file)
    
    def convert_to_spacy_format(self, output_path: str) -> None:
        """
        Converte il dataset in formato compatibile con spaCy.
        """
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        
        for _, row in self.df.iterrows():
            text = row["processed_text"]
            entities = self._extract_entities(row)
            
            doc = nlp.make_doc(text)
            ents = []
            occupied = set()
            for start, end, label in entities:
                if any(i in occupied for i in range(start, end)):
                    continue  # Evita sovrapposizioni
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
                    occupied.update(range(start, end))
            doc.ents = ents
            doc_bin.add(doc)
        
        doc_bin.to_disk(output_path)
    
    def _extract_entities(self, row) -> List[Tuple[int, int, str]]:
        """
        Estrai entità dal dataset con gli offset corretti.
        """
        entities = []
        attributes = {"Age": "AGE", "Sex": "SEX", "Symptoms": "SYMPTOM", "Disease": "DISEASE"}
        
        text = row["processed_text"]
        for column, label in attributes.items():
            values = str(row[column]).split(",")
            for value in values:
                value = value.strip()
                start = text.find(value)
                if start != -1:
                    entities.append((start, start + len(value), label))
        
        return entities

class NERTrainer:
    def __init__(self, model: str = "en_core_sci_sm"):
        self.nlp = spacy.load(model)
    
    def train(self, train_data_path: str, output_dir: str, epochs: int = 10):
        """
        Allena il modello spaCy sul dataset fornito.
        """
        doc_bin = DocBin().from_disk(train_data_path)
        optimizer = self.nlp.resume_training()
        
        for _ in range(epochs):
            for doc in doc_bin.get_docs(self.nlp.vocab):
                example = spacy.training.Example.from_dict(
                    doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
                )
                self.nlp.update([example], sgd=optimizer)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_dir)

if __name__ == "__main__":
    # ESEMPIO DI UTILIZZO
    data_processor = DataProcessor(r"C:\Users\batti\bioner\src\processed_data.csv")
    data_processor.convert_to_spacy_format(r"C:\Users\batti\bioner\models\scispacy_data\train.spacy")
    
    trainer = NERTrainer("en_core_sci_sm")
    trainer.train(r"C:\Users\batti\bioner\models\scispacy_data\train.spacy", "output_model")
