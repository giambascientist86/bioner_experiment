import pandas as pd
import re
import spacy

class TextProcessor:
    """Classe per la pulizia e il preprocessing del testo clinico."""

    def __init__(self):
        """Inizializza il processore con il modello spaCy clinico."""
        self.nlp = spacy.load("en_core_web_sm")  # Modello specifico per testi medici

    def clean_text(self, text: str) -> str:
        """
        Applica le operazioni di pulizia sul testo clinico.

        Args:
            text (str): Testo da pulire.

        Returns:
            str: Testo processato.
        """
        text = text.lower()  # Converti in minuscolo
        text = re.sub(r'\s+', ' ', text)  # Rimuove spazi multipli
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Rimuove caratteri speciali
        return text.strip()

    def tokenize_text(self, text: str) -> str:
        """
        Tokenizza e lemmatizza il testo clinico usando spaCy.

        Args:
            text (str): Testo pulito.

        Returns:
            str: Testo tokenizzato e lemmatizzato.
        """
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    def process_text(self, text: str) -> str:
        """
        Esegue l'intero preprocessing: pulizia, tokenizzazione, lemmatizzazione.

        Args:
            text (str): Testo originale.

        Returns:
            str: Testo preprocessato.
        """
        text = self.clean_text(text)
        return self.tokenize_text(text)

def main():
    """Legge il dataset e applica il preprocessing solo alla colonna 'text'."""
    dataset_path = r"C:\Users\batti\bioner\src\dataset_with_features.csv"  # Modifica se il path è diverso
    df = pd.read_csv(dataset_path)

    processor = TextProcessor()
    df["processed_text"] = df["text"].apply(processor.process_text)

    # Salva il dataset processato
    output_path = r"C:\Users\batti\bioner\src\processed_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessing completato. Dataset salvato in {output_path}")

if __name__ == "__main__":
    main()
