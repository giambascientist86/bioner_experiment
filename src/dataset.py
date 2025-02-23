import zipfile
import os
from pathlib import Path
from typing import List, Dict

class MACCROBATDataset:
    def __init__(self, zip_path: str, extract_path: str = "data/MACCROBAT_Extracted"):
        self.zip_path = zip_path
        self.extract_path = Path(extract_path)
        self.extract_path.mkdir(parents=True, exist_ok=True)  # Assicura che la directory esista
        self._extract_zip()

    def _extract_zip(self) -> None:
        """Extract the dataset zip file while handling empty filenames and invalid entries."""
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.strip() and not file_info.filename.endswith("/"):
                        zip_ref.extract(file_info, self.extract_path)
            print(f"Dataset extracted to {self.extract_path}")
        except zipfile.BadZipFile:
            raise ValueError("Invalid ZIP file. Please check the dataset.")

    def load_data(self) -> List[Dict[str, str]]:
        """Load the extracted text and annotation files into a structured format."""
        data = []
        txt_files = list(self.extract_path.glob("*.txt"))
        
        for txt_file in txt_files:
            ann_file = txt_file.with_suffix(".ann")
            if ann_file.exists():
                text = self._read_txt(txt_file)
                annotations = self._read_ann(ann_file)
                data.append({"filename": txt_file.name, "text": text, "annotations": annotations})
        
        return data

    def _read_txt(self, file_path: Path) -> str:
        """Read text from a .txt file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_ann(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse annotations from a .ann file, filtering only relevant entities."""
        relevant_entities = {"Age", "Sex", "Symptom", "Disease"}
        annotations = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    entity_info, entity_text = parts[1], parts[2]
                    entity_type = entity_info.split(" ")[0]
                    if entity_type in relevant_entities:
                        annotations.append({"type": entity_type, "text": entity_text})
        
        return annotations

# Esempio di utilizzo
dataset = MACCROBATDataset(zip_path=r"C:\Users\batti\bioner\data\MACCROBAT2020_Simplified.zip")
data = dataset.load_data()
print(f"Caricati {len(data)} record dal dataset.")