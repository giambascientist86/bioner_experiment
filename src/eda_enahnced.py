import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import random
from wordcloud import WordCloud
import dataset
from dataset import MACCROBATDataset

class MACCROBATEDA:
    """Classe per l'analisi esplorativa del dataset MACCROBAT"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = pd.DataFrame(dataset)

    def show_sample(self, num_samples=1):
        samples = random.sample(self.dataset, num_samples)
        for sample in samples:
            print(f"\n**Esempio di Testo (file: {sample['filename']})**\n")
            print(sample["text"][:500] + "...")
            print("\n**Annotazioni:**")
            for ann in sample["annotations"]:
                print(ann)

    def entity_distribution(self):
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
        self.df["text_length"] = self.df["text"].apply(len)
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["text_length"], bins=20, kde=True)
        plt.title("Distribuzione della Lunghezza dei Testi")
        plt.xlabel("Numero di Caratteri")
        plt.ylabel("Frequenza")
        plt.show()
    
    def annotation_length_distribution(self):
        ann_lengths = [len(ann["text"]) for entry in self.df["annotations"] for ann in entry]
        plt.figure(figsize=(8, 5))
        sns.histplot(ann_lengths, bins=20, kde=True)
        plt.title("Distribuzione della Lunghezza delle Annotazioni")
        plt.xlabel("Numero di Caratteri")
        plt.ylabel("Frequenza")
        plt.show()

    def co_occurrence_matrix(self):
        entity_types = [ann["type"] for entry in self.df["annotations"] for ann in entry]
        entity_df = pd.DataFrame(entity_types, columns=["Entity"])
        sns.heatmap(pd.crosstab(entity_df["Entity"], entity_df["Entity"]), annot=True, cmap="Blues")
        plt.title("Matrice di Co-occorrenza delle Entità")
        plt.show()

    def wordcloud_for_entity(self, entity_type):
        """
        Genera una word cloud per un'entità specifica (es. Symptoms, Age, Disease).
        
        :param entity_type: Il tipo di entità per cui generare la word cloud.
        """
        words = " ".join(
            ann["text"]
            for entry in self.df["annotations"]
            for ann in entry
            if ann["type"] == entity_type
        )
        
        if not words.strip():  # Controllo se la stringa è vuota
            print(f"\n⚠️ Nessuna annotazione trovata per l'entità '{entity_type}'!")
            return
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud per {entity_type}")
        plt.show()

    
    def age_distribution(self):
        ages = [int(ann["text"]) for entry in self.df["annotations"] for ann in entry if ann["type"] == "Age" and ann["text"].isdigit()]
        plt.figure(figsize=(8, 5))
        sns.histplot(ages, bins=10, kde=True)
        plt.title("Distribuzione delle Età nei Testi")
        plt.xlabel("Età")
        plt.ylabel("Frequenza")
        plt.show()
    
    def check_missing_annotations(self):
        missing_annotations = self.df[self.df["annotations"].apply(len) == 0]
        num_missing = len(missing_annotations)
        if num_missing > 0:
            print(f"\n⚠️ {num_missing} file senza annotazioni!")
            print(missing_annotations["filename"].tolist())
        else:
            print("\n✅ Nessun file senza annotazioni!")

# ESEMPIO DI UTILIZZO
if __name__ == "__main__":
    dataset = MACCROBATDataset(zip_path=r"C:\\Users\\batti\\bioner\\data\\MACCROBAT2020_Simplified.zip").load_data()
    eda = MACCROBATEDA(dataset)
    eda.show_sample(num_samples=3)
    eda.entity_distribution()
    eda.text_length_distribution()
    eda.annotation_length_distribution()
    eda.co_occurrence_matrix()
    eda.wordcloud_for_entity("Symptoms")
    eda.wordcloud_for_entity("Disease")
    eda.age_distribution()
    eda.check_missing_annotations()
