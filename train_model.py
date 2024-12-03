import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import nltk
import warnings

warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        # Inicializar componentes necesarios
        self.tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(kernel='rbf', probability=True)
        self.stemmer = SnowballStemmer('spanish')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))

    def preprocess_text(self, text):
        """Preprocesa el texto aplicando normalización básica."""
        if not isinstance(text, str):
            return ""
        # Convertir a minúsculas y eliminar caracteres especiales
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenización y eliminación de stopwords
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    # def get_bert_embedding(self, text):
    #     """Obtiene el embedding BERT para un texto dado."""
    #     inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     # Usar el embedding del token [CLS] como representación del texto
    #     return outputs.last_hidden_state[:, 0, :].numpy()
    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Usar la media de todos los embeddings
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def prepare_data(self, df):
        """Prepara los datos para el entrenamiento."""
        # Mapear preguntas a sentimientos
        sentiment_mapping = {
            1: 'alegria', 6: 'alegria',
            2: 'tristeza',
            3: 'estres', 9: 'estres',
            4: 'inquietud', 5: 'inquietud',
            7: 'miedo', 10: 'miedo',
            8: 'enojo'
        }
        
        # Preparar datos de entrenamiento
        X = []
        y = []
        
        for idx, row in df.iterrows():
            for q_num in range(1, 11):
                col_name = f"{q_num}. "
                cols = [col for col in df.columns if col.startswith(col_name)]
                if cols:
                    text = row[cols[0]]
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        embedding = self.get_bert_embedding(processed_text)
                        X.append(embedding[0])
                        y.append(sentiment_mapping[q_num])
        
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)
        
        print(X)
        return X, y

    def train_model(self, X, y):
        """Entrena el modelo de clasificación."""
        # División del dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenamiento del modelo
        self.classifier.fit(X_train, y_train)
        
        # Validación cruzada
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        
        # Evaluación en conjunto de prueba
        y_pred = self.classifier.predict(X_test)
        
        return {
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred)
        }

    def plot_results(self, results):
        """Visualiza los resultados del modelo."""
        # Matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Matriz de Confusión')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.savefig('matriz_confusion.png')
        plt.show()

        # Resultados de validación cruzada
        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_scores'])
        plt.title('Distribución de Puntuaciones en Validación Cruzada')
        plt.ylabel('Puntuación')
        plt.savefig('validacion_cruzada.png')
        plt.show()

    def predict_sentiment(self, text):
        """Predice el sentimiento para un nuevo texto."""
        processed_text = self.preprocess_text(text)
        embedding = self.get_bert_embedding(processed_text)
        prediction = self.classifier.predict(embedding)
        probabilities = self.classifier.predict_proba(embedding)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))

def main():
    # Cargar datos
    df = pd.read_csv('Textos_Dataset_Completo_utf8.csv')
    
    # Inicializar y entrenar el modelo
    analyzer = SentimentAnalyzer()
    X, y = analyzer.prepare_data(df)
    
    
    print(y)
    results = analyzer.train_model(X, y)
    
    # Visualizar resultados
    analyzer.plot_results(results)
    
    # Ejemplo de predicción
    text = "Me siento muy contento por haber logrado mis metas"
    sentiment, probs = analyzer.predict_sentiment(text)
    print(f"\nTexto de ejemplo: {text}")
    print(f"Sentimiento predicho: {sentiment}")
    print("\nProbabilidades por clase:")
    for sentiment, prob in probs.items():
        print(f"{sentiment}: {prob:.2f}")

if __name__ == "__main__":
    main()
