from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import torch
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import random

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class MultiRouteAugmenter:
    def __init__(self, batch_size=8, num_workers=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.translation_routes = [
            ('Helsinki-NLP/opus-mt-es-en', 'Helsinki-NLP/opus-mt-en-es'),  # español -> inglés -> español
            ('Helsinki-NLP/opus-mt-es-fr', 'Helsinki-NLP/opus-mt-fr-es'),  # español -> francés -> español
            ('Helsinki-NLP/opus-mt-es-it', 'Helsinki-NLP/opus-mt-it-es'),  # español -> italiano -> español
        ]
        
        print("Cargando modelos...")
        self.models = {}
        self.tokenizers = {}

        for route in self.translation_routes:
            for model_name in route:
                if model_name not in self.models:
                    self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
                    self.models[model_name] = MarianMTModel.from_pretrained(model_name).to(self.device)
                    self.models[model_name].eval()
        
        torch.set_grad_enabled(False)

    def translate(self, texts: List[str], source_model: str, target_model: str) -> List[str]:

        encoded = self.tokenizers[source_model](texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        

        outputs = self.models[source_model].generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            length_penalty=0.6,
            max_length=512
        )

        intermediate_texts = self.tokenizers[source_model].batch_decode(outputs, skip_special_tokens=True)
        
  
        back_encoded = self.tokenizers[target_model](intermediate_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        back_outputs = self.models[target_model].generate(
            input_ids=back_encoded['input_ids'].to(self.device),
            attention_mask=back_encoded['attention_mask'].to(self.device),
            num_beams=4,
            length_penalty=0.6,
            max_length=512
        )
        
        return self.tokenizers[target_model].batch_decode(back_outputs, skip_special_tokens=True)

    def augment_text(self, text: str) -> str:
        """Augmenta un solo texto usando una ruta de traducción aleatoria"""
        if pd.isna(text) or not text.strip():
            return text
            
        route = random.choice(self.translation_routes)
        return self.translate([text], route[0], route[1])[0]

    def augment_dataset(self, df: pd.DataFrame, question_cols: List[str]) -> pd.DataFrame:
        """Aumenta el dataset balanceadamente"""
        new_rows = []
        
        for idx, row in tqdm(df.iterrows(), desc="Procesando filas", total=len(df)):
 
            new_row = row.copy()
            
    
            valid_cols = [col for col in question_cols if pd.notna(row[col]) and row[col].strip()]
            
            if valid_cols:

                num_questions = random.randint(1, min(2, len(valid_cols)))
                cols_to_vary = random.sample(valid_cols, num_questions)

                for col in cols_to_vary:
                    new_row[col] = self.augment_text(row[col])
            
            new_rows.append(new_row)

        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

def main():
    BATCH_SIZE = 16
    NUM_WORKERS = 12
    
    augmenter = MultiRouteAugmenter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    df = pd.read_csv('dataset_normalizado_utf8.csv')
    question_cols = [f'pregunta_{i}' for i in range(1, 11)]
    
    augmented_df = augmenter.augment_dataset(df, question_cols)
    
    augmented_df.to_csv('dataset_normalizado_utf8_aumentado.csv', index=False)
    
    print(f"Dataset original: {len(df)} filas")
    print(f"Dataset aumentado: {len(augmented_df)} filas")

if __name__ == "__main__":
    main()