import os
import pandas as pd
from typing import List

class DataLoader:
    """
    Carica un file CSV specificato dal percorso.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Carica il file CSV dal percorso fornito.
        """
        return pd.read_csv(self.file_path)


class DataPreprocessor:
    """
    Preprocessa i dati del dataset per la conversione in formato CoNLL.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def group_by_filename(self) -> List[pd.DataFrame]:
        """
        Raggruppa il dataset per 'Filename' e restituisce una lista di gruppi.
        """
        return [group for _, group in self.data.groupby("filename")]

    def prepare_conll_format(self, group: pd.DataFrame) -> str:
        """
        Converte un gruppo di dati in formato CoNLL.
        """
        conll_output = []
        for _, row in group.iterrows():
            conll_output.append(f"{row['token']} {row['bio_tag']}")
        return "\n".join(conll_output)


class CoNLLWriter:
    """
    Gestisce la scrittura dei dati preprocessati in un file CoNLL.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path

    def write_to_file(self, grouped_data: List[pd.DataFrame], base_filename: str):
        """
        Scrive i dati preprocessati nel file CoNLL specificato dal percorso.
        """
        with open(self.output_path, "w") as f:
            for group in grouped_data:
                conll_data = DataPreprocessor(group).prepare_conll_format(group)
                f.write(conll_data + "\n\n")  # Riga vuota tra frasi


class NERPipeline:
    """
    Esegue l'intero processo di caricamento, preprocessing e scrittura del dataset in formato CoNLL.
    """
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        # Step 1: Carica il file CSV
        data_loader = DataLoader(self.input_path)
        df = data_loader.load_data()

        # Step 2: Preprocessa i dati (raggruppa per Filename)
        preprocessor = DataPreprocessor(df)
        grouped_data = preprocessor.group_by_filename()

        # Step 3: Scrivi il file CoNLL nel file di output
        base_filename = os.path.basename(self.input_path).split('.')[0]  # Estrae il nome base del file CSV
        conll_writer = CoNLLWriter(self.output_path)
        conll_writer.write_to_file(grouped_data, base_filename)


# Esecuzione della pipeline con input_path e output_path come file individuali
input_path = r"C:\Users\batti\bioner\src\token_level_dataset.csv"  # Percorso del file CSV di input
output_path = "./ner_dataset.conll"  # Percorso del file CoNLL di output

pipeline = NERPipeline(input_path, output_path)
pipeline.run()
