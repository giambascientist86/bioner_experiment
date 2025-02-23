import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from typing import List, Tuple

class DataProcessor:
    def __init__(self, input_file: str):
        """
        Inizializza la classe DataProcessor con il percorso del file CSV.
        """
        self.input_file = input_file
        self.df = pd.read_csv(input_file)
    
    def convert_to_spacy_format(self, output_train_path: str, output_test_path: str, test_size: float = 0.2) -> None:
        """
        Converte il dataset in formato compatibile con spaCy e divide i dati in training e test set.
        """
        nlp = spacy.blank("en")
        
        # Dividi il dataset in training e test set
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=42)
        
        # Crea due DocBin per il training e il test
        train_doc_bin = DocBin()
        test_doc_bin = DocBin()
        
        # Converte il set di addestramento in formato spaCy
        self._convert_set_to_spacy_format(train_df, train_doc_bin, nlp)
        train_doc_bin.to_disk(output_train_path)

        # Converte il set di test in formato spaCy
        self._convert_set_to_spacy_format(test_df, test_doc_bin, nlp)
        test_doc_bin.to_disk(output_test_path)

    def _convert_set_to_spacy_format(self, df, doc_bin, nlp):
        """
        Converte il DataFrame in formato spaCy DocBin.
        """
        for _, row in df.iterrows():
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
    def __init__(self):
        """
        Inizializza il trainer NER.
        """
        self.model = None
    
    def train(self, train_data_path: str, output_dir: str, epochs: int = 10) -> None:
        """
        Allena il modello NER con i dati di addestramento.
        """
        nlp = spacy.blank("en")  # Crea un modello vuoto spaCy
        db = DocBin().from_disk(train_data_path)
        
        # Aggiungi componente NER al pipeline
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

        # Aggiungi le etichette
        for doc in db.get_docs():
            for ent in doc.ents:
                ner.add_label(ent.label_)
        
        # Addestra il modello
        optimizer = nlp.begin_training()
        for epoch in range(epochs):
            for batch in spacy.util.minibatch(db.get_docs(), size=8):
                for doc in batch:
                    example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
                    nlp.update([example], drop=0.5)
        
        # Salva il modello addestrato
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir_path)

    def load_model(self, model_dir: str) -> None:
        """
        Carica un modello NER pre-esistente.
        """
        self.model = spacy.load(model_dir)


class NEREvaluator:
    def __init__(self, model_dir: str):
        """
        Inizializza il valutatore NER.
        """
        self.model = spacy.load(model_dir)
    
    def evaluate(self, test_data_path: str) -> Tuple[float, float, float]:
        """
        Valuta il modello NER sui dati di test.
        """
        db = DocBin().from_disk(test_data_path)
        y_true = []
        y_pred = []

        for doc in db.get_docs():
            true_entities = {ent.label_: (ent.start_char, ent.end_char) for ent in doc.ents}
            predicted_entities = {ent.label_: (ent.start_char, ent.end_char) for ent in self.model(doc).ents}

            for label in true_entities:
                y_true.append(label in true_entities)
                y_pred.append(label in predicted_entities)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        return precision, recall, f1


if __name__ == "__main__":
    # Preprocessing dei dati e salvataggio nei file .spacy
    data_processor = DataProcessor(r"C:\Users\batti\bioner\src\processed_data.csv")
    data_processor.convert_to_spacy_format(
        r"C:\Users\batti\bioner\models\scispacy_data\train.spacy", 
        r"C:\Users\batti\bioner\models\scispacy_data\test.spacy"
    )

    # Allena il modello con i dati di addestramento
    trainer = NERTrainer()
    trainer.train(r"C:\Users\batti\bioner\models\scispacy_data\train.spacy", "output_model", epochs=10)

    # Valuta il modello sui dati di test
    evaluator = NEREvaluator("output_model")
    precision, recall, f1 = evaluator.evaluate(r"C:\Users\batti\bioner\models\scispacy_data\test.spacy")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
