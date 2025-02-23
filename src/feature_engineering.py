import pandas as pd

class FeatureEngineer:
    """
    Classe per il Feature Engineering del dataset estratto dal sistema di Named Entity Recognition.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """Inizializza con il dataset estratto."""
        self.df = dataframe.copy()
    
    def calculate_symptom_length(self):
        """Calcola la lunghezza media dei sintomi per ogni paziente."""
        self.df['Avg_Symptom_Length'] = self.df['Symptoms'].apply(
            lambda x: sum(len(symptom) for symptom in x.split(';')) / len(x.split(';')) if x else 0
        )
    
    def count_entities(self):
        """Conta il numero di sintomi e malattie per paziente."""
        self.df['Num_Symptoms'] = self.df['Symptoms'].apply(lambda x: len(x.split(';')) if x else 0)
        self.df['Num_Diseases'] = self.df['Disease'].fillna("").astype(str).apply(lambda x: len(x.split(';')))

    
    def multiple_diseases_flag(self):
        """Crea una feature booleana per la presenza di più malattie."""
        self.df['Multiple_Diseases'] = self.df['Num_Diseases'].apply(lambda x: 1 if x > 1 else 0)
    
    def symptom_disease_ratio(self):
        """Calcola il rapporto tra sintomi e malattie."""
        self.df['Symptom_Disease_Ratio'] = self.df.apply(
            lambda row: row['Num_Symptoms'] / row['Num_Diseases'] if row['Num_Diseases'] > 0 else 0,
            axis=1
        )
    
    def generate_features(self):
        """Esegue tutti i metodi per generare le feature."""
        self.calculate_symptom_length()
        self.count_entities()
        self.multiple_diseases_flag()
        self.symptom_disease_ratio()
        return self.df

# Esempio di utilizzo
if __name__ == "__main__":
    # Carica il dataset estratto
    df = pd.read_csv(r"C:\Users\batti\bioner\src\extracted_data.csv")
    
    # Inizializza il Feature Engineer
    fe = FeatureEngineer(df)
    df_features = fe.generate_features()
    
    # Salva il dataset con le nuove feature
    df_features.to_csv("dataset_with_features.csv", index=False)
    print("Feature Engineering completato! Il dataset è stato salvato con successo.")
