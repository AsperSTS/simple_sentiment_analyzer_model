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
import datetime
import json
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        # Inicializar componentes necesarios
        # self.tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        # self.model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.pretrained_model_name = "PlanTL-GOB-ES/roberta-base-bne"
        self.tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        self.model = AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(kernel='linear' , probability=True)
        self.stemmer = SnowballStemmer('spanish')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))
        self.experiment_dir = self.create_experiment_directory()
    def create_experiment_directory(self):
        """
        Crea un nuevo directorio para el experimento actual.
        """
        base_dir = 'experiments'
        os.makedirs(base_dir, exist_ok=True)
        
        # Encontrar el siguiente número de ejecución
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run')]
        if not existing_runs:
            next_run = 1
        else:
            run_numbers = [int(run.replace('run', '')) for run in existing_runs]
            next_run = max(run_numbers) + 1
            
        # Crear directorio para esta ejecución
        run_dir = os.path.join(base_dir, f'run{next_run}')
        os.makedirs(run_dir)
        
        return run_dir
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
        
        # Mapeo de números de pregunta a nombres de columna en el CSV
        column_mapping = {
            1: "1. Describa, ¿en qué situaciones últimamente ha sentido alegría?",
            2: "2. Especifique, ¿en qué situaciones últimamente ha sentido ganas de llorar?",
            3: "3. En las últimas dos semanas, ¿en qué momentos se ha sentido cansado?",
            4: "4. ¿En qué situaciones de su día a día, puede identificar que se ha sentido preocupado?",
            5: "5. Cuando la preocupación se hace presente en su vida, ¿cuáles son las sensaciones corporales que experimenta?",
            6: "6. Si su escritor favorito le pidiera que le ayude a buscar el significado de la felicidad, ¿cuál sería la idea que usted escribiría?",
            7: "7. Cuando tiene una elección importante que hacer, mencione ¿cuáles serían los pasos que seguiría para llegar a tomar una decisión?",
            8: "8. Cuando sucede algo inesperado en su vida, por ejemplo: llegar tarde al trabajo, el automóvil se descompuso, se quedó sin batería el celular, olvidar las llaves, etc., describa ¿cómo reacciona ante estas situaciones?",
            9: "9. Describa ¿En qué situaciones de su vida considera que se siente estresado o molesto?",
            10: "10. Por favor, describa brevemente ¿qué hace cuando tiene que enfrentar una situación difícil?"
        }
        
        for idx, row in df.iterrows():
            for q_num, column_name in column_mapping.items():
                if column_name in df.columns:
                    text = row[column_name]
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        embedding = self.get_bert_embedding(processed_text)
                        X.append(embedding[0])
                        y.append(sentiment_mapping[q_num])
        
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
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
        plt.savefig(os.path.join(self.experiment_dir,f'matriz_confusion_{counter}.png'))
        plt.show()

        # Resultados de validación cruzada
        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_scores'])
        plt.title('Distribución de Puntuaciones en Validación Cruzada')
        plt.ylabel('Puntuación')
        plt.savefig(os.path.join(self.experiment_dir,f'validacion_cruzada_{counter}.png'))
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
    def save_experiment_metrics(self, results, execution_time):
        """
        Guarda las métricas y parámetros del experimento en un archivo JSON.
        """
        print("Guardando métricas del experimento...")
        
        # Convertir el classification report de string a diccionario de una manera más robusta
        classification_dict = {}
        report_lines = results['classification_report'].split('\n')
        for line in report_lines:
            if line and not line.startswith('micro') and not line.startswith('macro') and not line.startswith('weighted') and not line.startswith('accuracy'):
                line_parts = line.strip().split()
                if len(line_parts) >= 4:  # Asegurarse de que la línea tiene suficientes elementos
                    class_name = line_parts[0]
                    try:
                        classification_dict[class_name] = {
                            'precision': float(line_parts[1]),
                            'recall': float(line_parts[2]),
                            'f1-score': float(line_parts[3]),
                            'support': int(line_parts[4]) if len(line_parts) > 4 else 0
                        }
                    except (ValueError, IndexError):
                        continue  # Saltarse líneas que no pueden ser parseadas correctamente

        metrics_data = {
            'experiment_id': os.path.basename(self.experiment_dir),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_parameters': {
                'embedding_model': self.pretrained_model_name,
                'classifier': 'SVM',
                # 'kernel': self.classifier.kernel,
                # 'probability': self.classifier.probability,
            },
            'preprocessing_parameters': {
                'stemmer': 'SnowballStemmer-spanish',
                'remove_stopwords': True,
                'max_length': 512,
            },
            'performance_metrics': {
                'cross_validation_scores': {
                    'mean': float(np.mean(results['cv_scores'])),
                    'std': float(np.std(results['cv_scores'])),
                    'scores': results['cv_scores'].tolist()
                },
                'classification_metrics': classification_dict,
                'confusion_matrix': results['confusion_matrix'].tolist()
            },
            'execution_metrics': {
                'total_execution_time': execution_time,
            }
        }
        
        # Guardar métricas en archivo JSON
        metrics_file = os.path.join(self.experiment_dir, 'experiment_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=4)
        
        print(f"Métricas guardadas en: {metrics_file}")
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
    
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
    
    count = analyzer.contar_pk1_mas_uno()

    # Guardar métricas del experimento
    analyzer.save_experiment_metrics(results, execution_time)
    
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
    with open(os.path.join(analyzer.experiment_dir,f'sentiment_model_{count}.pkl'), 'wb') as f:
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
