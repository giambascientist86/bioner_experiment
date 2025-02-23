import os
import csv

class AnnotationParser:
    """Classe per il parsing dei file .ann e l'estrazione delle informazioni chiave."""
    
    def __init__(self, ann_path):
        self.ann_path = ann_path

    def parse(self):
        """Parsa il file .ann e restituisce le informazioni estratte."""
        symptoms = []
        diseases = []
        age = None
        sex = None
        
        with open(self.ann_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue

                entity_info, entity_text = parts[1], parts[2]
                entity_type = entity_info.split()[0]

                if entity_type == "Age":
                    age = entity_text
                elif entity_type == "Sex":
                    sex = entity_text
                elif entity_type == "Sign_symptom":
                    symptoms.append(entity_text)
                elif entity_type == "Disease_disorder":
                    diseases.append(entity_text)

        return PatientRecord(age, sex, symptoms, diseases)


class PatientRecord:
    """Classe che rappresenta un paziente con le informazioni estratte."""

    def __init__(self, age, sex, symptoms, diseases):
        self.age = age
        self.sex = sex
        self.symptoms = ", ".join(symptoms) if symptoms else ""
        self.diseases = ", ".join(diseases) if diseases else ""

    def to_dict(self, filename, text):
        """Converte i dati in un dizionario per il salvataggio CSV."""
        return {
            "filename": filename,
            "text": text,
            "Age": self.age,
            "Sex": self.sex,
            "Symptoms": self.symptoms,
            "Disease": self.diseases,
        }


class DataProcessor:
    """Classe per la gestione del dataset e il salvataggio in CSV."""

    def __init__(self, data_dir, output_csv):
        self.data_dir = data_dir
        self.output_csv = output_csv

    def process_data(self):
        """Legge i file di testo e annesse annotazioni, quindi salva i dati estratti in un CSV."""
        data = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_id = filename.split(".")[0]
                txt_path = os.path.join(self.data_dir, filename)
                ann_path = os.path.join(self.data_dir, f"{file_id}.ann")

                if os.path.exists(ann_path):
                    with open(txt_path, "r", encoding="utf-8") as file:
                        text = file.read().strip()

                    patient_record = AnnotationParser(ann_path).parse()
                    data.append(patient_record.to_dict(filename, text))

        self._save_to_csv(data)

    def _save_to_csv(self, data):
        """Salva i dati estratti in un file CSV."""
        fieldnames = ["filename", "text", "Age", "Sex", "Symptoms", "Disease"]
        
        with open(self.output_csv, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    data_dir = "C:\Users\batti\bioner\src\data\MACCROBAT_Extracted"
    output_csv = "extracted_data.csv"

    processor = DataProcessor(data_dir, output_csv)
    processor.process_data()
    print(f"Dati estratti e salvati in {output_csv}")
