import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import random
import dataset
from dataset import MACCROBATDataset

class MACCROBATEDA:
    """Classe per l'analisi esplorativa del dataset MACCROBAT"""
    
    def __init__(self, dataset):
        """
        Inizializza l'EDA con il dataset fornito.

        :param dataset: Lista di dizionari contenenti "filename", "text" e "annotations".
        """
        self.dataset = dataset
        self.df = pd.DataFrame(dataset)

    def show_sample(self, num_samples=1):
        """
        Mostra un esempio casuale dal dataset.
        
        :param num_samples: Numero di esempi da visualizzare.
        """
        samples = random.sample(self.dataset, num_samples)
        for sample in samples:
            print(f"\n**Esempio di Testo (file: {sample['filename']})**\n")
            print(sample["text"][:500] + "...")
            print("\n**Annotazioni:**")
            for ann in sample["annotations"]:
                print(ann)

    def entity_distribution(self):
        """
        Mostra la distribuzione delle entità annotate nel dataset.
        """
        entity_counts = Counter(
            [ann["type"] for entry in self.df["annotations"] for ann in entry]
        )
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()))
        plt.title("Distribuzione delle Entità Annotate")
        plt.ylabel("Frequenza")
        plt.xlabel("Tipo di Entità")
        plt.show()

    def text_length_distribution(self):
        """
        Analizza la distribuzione della lunghezza dei testi.
        """
        self.df["text_length"] = self.df["text"].apply(len)

        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["text_length"], bins=20, kde=True)
        plt.title("Distribuzione della Lunghezza dei Testi")
        plt.xlabel("Numero di Caratteri")
        plt.ylabel("Frequenza")
        plt.show()

    def check_missing_annotations(self):
        """
        Controlla se ci sono annotazioni mancanti nei file.
        """
        missing_annotations = self.df[self.df["annotations"].apply(len) == 0]
        num_missing = len(missing_annotations)

        if num_missing > 0:
            print(f"\n⚠️ {num_missing} file senza annotazioni!")
            print(missing_annotations["filename"].tolist())
        else:
            print("\n✅ Nessun file senza annotazioni!")

# ESEMPIO DI UTILIZZO
if __name__ == "__main__":
    # Caricare il dataset (assumendo che dataset_loader lo restituisca)
    dataset = MACCROBATDataset(zip_path=r"C:\Users\batti\bioner\data\MACCROBAT2020_Simplified.zip").load_data()

    eda = MACCROBATEDA(dataset)
    eda.show_sample(num_samples=3)  # Visualizza 2 esempi
    eda.entity_distribution()  # Analizza la distribuzione delle entità
    eda.text_length_distribution()  # Distribuzione della lunghezza dei testi
    eda.check_missing_annotations()  # Controllo annotazioni mancanti
