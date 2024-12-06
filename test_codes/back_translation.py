from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
import random

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class FastTextAugmenter:
    def __init__(self, batch_size=8, num_workers=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print("Cargando modelos...")
        self.model_name = 'Helsinki-NLP/opus-mt-es-en'
        self.back_model_name = 'Helsinki-NLP/opus-mt-en-es'
        
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.back_tokenizer = MarianTokenizer.from_pretrained(self.back_model_name)
        
        self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
        self.back_model = MarianMTModel.from_pretrained(self.back_model_name).to(self.device)
        
        self.model.eval()
        self.back_model.eval()
        torch.set_grad_enabled(False)

    def process_batch(self, batch_texts: List[str]) -> List[str]:
        """Procesa un lote de textos de manera eficiente"""
        encoded = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            length_penalty=0.6,
            max_length=512,
            return_dict_in_generate=True,
            output_scores=False
        )
        
        eng_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        back_encoded = self.back_tokenizer(eng_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        back_input_ids = back_encoded['input_ids'].to(self.device)
        back_attention_mask = back_encoded['attention_mask'].to(self.device)
        
        back_outputs = self.back_model.generate(
            input_ids=back_input_ids,
            attention_mask=back_attention_mask,
            num_beams=4,
            length_penalty=0.6,
            max_length=512
        )
        
        return self.back_tokenizer.batch_decode(back_outputs, skip_special_tokens=True)

    def augment_dataset(self, df: pd.DataFrame, question_cols: List[str]) -> pd.DataFrame:
        """Aumenta el dataset generando una variación por fila"""
        
        # Seleccionar aleatoriamente una pregunta por fila para variar
        new_rows = []
        
        for idx, row in df.iterrows():
            # Crear una copia de la fila original
            new_row = row.copy()
            
            # Filtrar las columnas de pregunta que tienen contenido
            valid_cols = [col for col in question_cols if pd.notna(row[col])]
            
            if valid_cols:
                # Seleccionar aleatoriamente una pregunta para variar
                col_to_vary = random.choice(valid_cols)
                
                # Procesar solo esa pregunta
                variation = self.process_batch([row[col_to_vary]])[0]
                new_row[col_to_vary] = variation
            
            new_rows.append(new_row)
        
        # Combinar dataset original con variaciones
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

def main():
    BATCH_SIZE = 16
    NUM_WORKERS = 12
    
    augmenter = FastTextAugmenter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    df = pd.read_csv('dataset_normalizado_utf8.csv')
    question_cols = [f'pregunta_{i}' for i in range(1, 11)]
    
    augmented_df = augmenter.augment_dataset(df, question_cols)
    
    augmented_df.to_csv('dataset_normalizado_utf8_aumentado_optimizado.csv', index=False)
    
    print(f"Dataset original: {len(df)} filas")
    print(f"Dataset aumentado: {len(augmented_df)} filas")

if __name__ == "__main__":
    main()