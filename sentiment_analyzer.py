import pickle
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
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from spacy import load
import time 
import os
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        # Inicializar componentes necesarios
        # self.tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        # self.model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        self.model = AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(kernel='rbf', probability=True)
        self.stemmer = SnowballStemmer('spanish')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))

    def preprocess_text(self, text):
        print("Preprocesando el texto...")
        """Preprocesa el texto aplicando normalización básica."""
        if not isinstance(text, str):
            return ""
        
        # # Convertir a minúsculas y eliminar caracteres especiales
        # text = text.lower()
        # text = re.sub(r'[^\w\s]', '', text)
        # # Tokenización y eliminación de stopwords
        # tokens = word_tokenize(text)
        # tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        # return ' '.join(tokens)
        # Añadir más pasos de limpieza
        
        text = re.sub(r'[^\w\sáéíóúñü]', '', text)  # Mantener acentos y ñ
        text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
        # Lematización en lugar de stemming para mantener mejor el significado
        nlp = load('es_core_news_sm')
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        
        return ' '.join(tokens)

    # def get_bert_embedding(self, text):
    #     """Obtiene el embedding BERT para un texto dado."""
    #     inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     # Usar el embedding del token [CLS] como representación del texto
    #     return outputs.last_hidden_state[:, 0, :].numpy()
    def get_bert_embedding(self, text):
        print("Obteniendo embedding BERT...")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Usar la media de todos los embeddings
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def prepare_data(self, df):
        print(f"Preparando datos para el entrenamiento...")
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
        
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        # print(X_balanced)
        return X_balanced, y_balanced

    def train_model(self, X, y):
        print("Entrenando modelo...")
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
    def contar_pk1_mas_uno(self, directorio='.'):
        """
        Cuenta el número de archivos .pk1 en un directorio y le suma 1.

        Args:
            directorio: La ruta al directorio que se va a examinar.

        Returns:
            El número de archivos .pk1 en el directorio + 1.
        """
        contador = 0
        for archivo in os.listdir(directorio):
            if archivo.endswith(".pkl"):
                contador += 1
        return contador + 1
    def plot_results(self, results, counter):
        print("Visualizando resultados...")
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
        plt.savefig(f'matriz_confusion_{counter}.png')
        plt.show()

        # Resultados de validación cruzada
        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_scores'])
        plt.title('Distribución de Puntuaciones en Validación Cruzada')
        plt.ylabel('Puntuación')
        plt.savefig(f'validacion_cruzada_{counter}.png')
        plt.show()

    def predict_sentiment(self, text):
        print("Prediciendo sentimiento...")
        """Predice el sentimiento para un nuevo texto."""
        processed_text = self.preprocess_text(text)
        embedding = self.get_bert_embedding(processed_text)
        prediction = self.classifier.predict(embedding)
        probabilities = self.classifier.predict_proba(embedding)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))
    def perform_eda(self, df):
        print("Realizando análisis exploratorio de datos...")
        """Realizar análisis exploratorio de datos."""
        # Distribución de sentimientos
        sentiment_counts = df['sentiment'].value_counts()
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(kind='bar')
        plt.title('Distribución de Sentimientos')
        plt.savefig('distribucion_sentimientos.png')
        
        for sentiment in df['sentiment'].unique():
            text = ' '.join(df[df['sentiment'] == sentiment]['text'])
            wordcloud = WordCloud().generate(text)
            plt.figure()
            plt.imshow(wordcloud)
            plt.title(f'Palabras frecuentes - {sentiment}')
            plt.savefig(f'wordcloud_{sentiment}.png')

def main():
    start_time = time.time()
    # Cargar datos
    df = pd.read_csv('Textos_Dataset_Completo_utf8.csv')
    analyzer = SentimentAnalyzer()
    
    # analyzer.perform_eda(df)
    
    # Inicializar y entrenar el modelo
    X, y = analyzer.prepare_data(df)
    
    
    print(y)
    results = analyzer.train_model(X, y)
    
    end_time = time.time()
    
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
    
    count = analyzer.contar_pk1_mas_uno()

    # Visualizar resultados
    analyzer.plot_results(results, count)
    
    # Lista de textos para testear el modelo
    texts = [
        "Me siento muy contento por haber logrado mis metas",
        "Estoy muy triste, no sé qué hacer",
        "Hoy es un buen día, estoy emocionado",
        "No me siento bien, creo que estoy enfermo",
        "Estoy muy enojado por lo que sucedió",
        "Estoy muy emocionado de empezar mi nuevo trabajo"
    ]
    components = {
        'classifier': analyzer.classifier,
        'label_encoder': analyzer.label_encoder,
        'tokenizer': analyzer.tokenizer,
        'model': analyzer.model
    }
    with open(f'sentiment_model_{count}.pkl', 'wb') as f:
        pickle.dump(components, f)
    
    # Recorrer la lista de textos para obtener las predicciones
    for text in texts:
        sentiment, probs = analyzer.predict_sentiment(text)
        print(f"\nTexto de ejemplo: {text}")
        print(f"Sentimiento predicho: {sentiment}")
        print("\nProbabilidades por clase:")
        for sentiment_class, prob in probs.items():
            print(f"{sentiment_class}: {prob:.2f}")
        

if __name__ == "__main__":
    main()
