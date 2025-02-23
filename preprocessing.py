import pandas as pd
from datasets import Dataset

def convert_csv_to_sentence_level(csv_path):
    """
    Converte un dataset in formato token-based in un dataset sentence-level.
    """
    df = pd.read_csv(csv_path)
    
    grouped_sentences = []
    grouped_labels = []

    current_sentence = []
    current_labels = []
    previous_filename = None

    for _, row in df.iterrows():
        filename, token, bio_tag = row["filename"], row["token"], row["bio_tag"]

        if previous_filename is None:
            previous_filename = filename

        if filename != previous_filename:
            # Aggiungi la frase e i relativi label alla lista finale
            grouped_sentences.append(" ".join(current_sentence))
            grouped_labels.append(current_labels)

            # Reset per la nuova frase
            current_sentence = []
            current_labels = []
            previous_filename = filename

        current_sentence.append(token)
        current_labels.append(bio_tag)

    # Aggiungi l'ultima frase
    if current_sentence:
        grouped_sentences.append(" ".join(current_sentence))
        grouped_labels.append(current_labels)

    # Creazione DataFrame con testo intero e labels
    sentence_df = pd.DataFrame({"text": grouped_sentences, "labels": grouped_labels})

    return sentence_df

# Esempio di utilizzo
csv_path = r"C:\Users\batti\bioner\src\token_level_dataset.csv"  # Sostituisci con il percorso corretto
sentence_df = convert_csv_to_sentence_level(csv_path)

# Converti in formato Dataset di Hugging Face
dataset = Dataset.from_pandas(sentence_df)

# Salva in CSV se necessario
sentence_df.to_csv(r".\processed_dataset_clinicalbert.csv", index=False)

print("✅ Conversione completata! Esempio di output:")
print(sentence_df.head())
