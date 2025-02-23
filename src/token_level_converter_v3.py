import csv
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class BioTagger:
    """Classe per convertire il dataset in formato token-level con tag BIO."""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.bio_tags = []

    def _get_bio_tags(self, text, age, sex, symptoms, diseases):
        """Genera i tag BIO per il testo."""
        tokens = word_tokenize(text)
        tagged_tokens = []

        # Aggiungi le entità con i tag BIO
        age_entities = self._generate_bio_tags(tokens, age, "Age")
        sex_entities = self._generate_bio_tags(tokens, sex, "Sex")
        symptom_entities = self._generate_bio_tags(tokens, symptoms, "Symptom")
        disease_entities = self._generate_bio_tags(tokens, diseases, "Disease")

        # Combina le etichette BIO
        for token in tokens:
            tag = 'O'  # Default tag for non-entity tokens (Outside)
            if token in age_entities:
                tag = 'B-Age' if age_entities[token] == 0 else 'I-Age'
            elif token in sex_entities:
                tag = 'B-Sex' if sex_entities[token] == 0 else 'I-Sex'
            elif token in symptom_entities:
                tag = 'B-Symptom' if symptom_entities[token] == 0 else 'I-Symptom'
            elif token in disease_entities:
                tag = 'B-Disease' if disease_entities[token] == 0 else 'I-Disease'
            tagged_tokens.append((token, tag))
        
        return tagged_tokens

    def _generate_bio_tags(self, tokens, entities, entity_type):
        """Genera un dizionario di entità per il tagging BIO."""
        entity_dict = {}
        if entities:
            for i, entity in enumerate(entities.split(", ")):
                for token in tokens:
                    if entity.lower() in token.lower():  # Verifica se il token contiene l'entità
                        entity_dict[token] = i
        return entity_dict

    def convert_to_bio(self, output_file):
        """Converte il dataset in formato token-level con BIO-tag e lo salva."""
        with open(self.data_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            with open(output_file, 'w', encoding='utf-8', newline='') as bio_file:
                fieldnames = ['filename', 'token', 'bio_tag']
                writer = csv.DictWriter(bio_file, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    filename = row['filename']
                    text = row['text']
                    age = row['Age']
                    sex = row['Sex']
                    symptoms = row['Symptoms']
                    diseases = row['Disease']
                    
                    tagged_tokens = self._get_bio_tags(text, age, sex, symptoms, diseases)
                    
                    for token, bio_tag in tagged_tokens:
                        writer.writerow({
                            'filename': filename,
                            'token': token,
                            'bio_tag': bio_tag
                        })

        print(f"File BIO-tagged salvato come {output_file}")

if __name__ == "__main__":
    input_file = r"C:\Users\batti\bioner\src\extracted_data.csv"  # Il dataset estratto
    output_file = "token_level_bio.csv"  # Il file di output con i token e i BIO tag
    
    tagger = BioTagger(input_file)
    tagger.convert_to_bio(output_file)
