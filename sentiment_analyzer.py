import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import nltk
import warnings
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from spacy import load
import time
import os
import datetime
import json
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        # Inicializar componentes necesarios
        self.label_encoder = LabelEncoder()
        self.svm_c_parameter = 1.0
        self.svm_gamma_parameter = 'scale'
        self.svm_kernel_parameter = 'sigmoid'
        self.classifier = SVC(kernel=self.svm_kernel_parameter, probability=True, 
                              C=self.svm_c_parameter, gamma=self.svm_gamma_parameter)
        self.stemmer = SnowballStemmer('spanish')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))
        self.experiment_dir = self.create_experiment_directory()

    def create_experiment_directory(self):
        """Crea un nuevo directorio para el experimento actual."""
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
        """Preprocesa el texto aplicando normalización y lematización."""
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'[^\w\sáéíóúñü]', '', text)  # Mantener acentos y ñ
        text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
        
        # Lematización 
        nlp = load('es_core_news_sm')
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        
        return ' '.join(tokens)
    #FUCION TF-IDF
    def tfidf_features(self, df, column_mapping):
        """
        Genera características TF-IDF para las respuestas a las preguntas.

        Args:
          df: DataFrame de pandas con las respuestas a las preguntas.
          column_mapping: Diccionario que mapea los números de pregunta 
                         a los nombres de columna en el DataFrame.

        Returns:
          Una matriz numpy con las características TF-IDF.
        """
        corpus = []
        for idx, row in df.iterrows():
            for q_num, column_name in column_mapping.items():
                if column_name in df.columns:
                    text = row[column_name]
                    corpus.append(text)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        return tfidf_matrix.toarray()

    def prepare_data(self, df):
        """Prepara los datos para el entrenamiento."""
        print(f"Preparando datos para el entrenamiento...")
        # Mapear preguntas a sentimientos
        sentiment_mapping = {
            1: 'alegria', 6: 'alegria',
            2: 'tristeza',
            3: 'estres', 9: 'estres',
            4: 'inquietud', 5: 'inquietud',
            7: 'miedo', 10: 'miedo',
            8: 'enojo'
        }
        
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
        
        # Generar características TF-IDF
        tfidf_matrix = self.tfidf_features(df, column_mapping)

        # Preparar datos de entrenamiento
        X = []
        y = []

        for idx, row in df.iterrows():
            for q_num, column_name in column_mapping.items():
                if column_name in df.columns:
                    text = row[column_name]
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        # Usar TF-IDF como características
                        X.append(tfidf_matrix[idx]) 
                        y.append(sentiment_mapping[q_num])

        X = np.array(X)
        y = self.label_encoder.fit_transform(y)

        # Experimentar con diferentes técnicas de balanceo de clases
        # smote = SMOTE(random_state=42)
        # X_balanced, y_balanced = smote.fit_resample(X, y)

        # oversampler = RandomOverSampler(random_state=42)
        # X_balanced, y_balanced = oversampler.fit_resample(X, y)

        undersampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)

        return X_balanced, y_balanced

    def train_model(self, X, y):
        """Entrena el modelo de clasificación."""
        print("Entrenando modelo...")
        # División del dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Definir espacio de búsqueda de hiperparámetros
        param_grid = {
            'C': [0.1, 1, 10], 
            'gamma': [1, 0.1, 0.01], 
            'kernel': ['linear', 'rbf', 'sigmoid']
        }

        # Realizar búsqueda de cuadrícula
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)

        # Mejor modelo encontrado
        self.classifier = grid.best_estimator_

        # Validación cruzada
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)

        # Evaluación en conjunto de prueba
        y_pred = self.classifier.predict(X_test)
        
        return {
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted')
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
        """Visualiza los resultados del modelo."""
        print("Visualizando resultados...")
        # Matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                    annot=True, 
                    fmt='d',
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Matriz de Confusión - kernel: {self.svm_kernel_parameter} - '
                  f'gamma: {self.svm_gamma_parameter} - c: {self.svm_c_parameter}')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.savefig(os.path.join(self.experiment_dir, f'matriz_confusion_{counter}.png'))
        plt.show()

        # Resultados de validación cruzada
        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_scores'])
        plt.title(f'Validación Cruzada - kernel: {self.svm_kernel_parameter} - '
                  f'gamma: {self.svm_gamma_parameter} - c: {self.svm_c_parameter}')
        plt.ylabel('Puntuación')
        plt.savefig(os.path.join(self.experiment_dir, f'validacion_cruzada_{counter}.png'))
        plt.show()

    def predict_sentiment(self, text):
        """Predice el sentimiento para un nuevo texto."""
        print("Prediciendo sentimiento...")
        processed_text = self.preprocess_text(text)
        
        # Generar características TF-IDF para el nuevo texto
        tfidf_vectorizer = TfidfVectorizer() 
        tfidf_vectorizer.fit([processed_text]) # Ajustar el vectorizador al nuevo texto
        features = tfidf_vectorizer.transform([processed_text]).toarray()
        
        prediction = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))

    def perform_eda(self, df):
        """Realizar análisis exploratorio de datos."""
        print("Realizando análisis exploratorio de datos...")
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
        """Guarda las métricas y parámetros del experimento en un archivo JSON."""
        print("Guardando métricas del experimento...")
        
        # Convertir el classification report de string a diccionario
        classification_dict = {}
        report_lines = results['classification_report'].split('\n')
        for line in report_lines:
            if line and not line.startswith('micro') and not line.startswith('macro') and \
               not line.startswith('weighted') and not line.startswith('accuracy'):
                line_parts = line.strip().split()
                if len(line_parts) >= 4: 
                    class_name = line_parts[0]
                    try:
                        classification_dict[class_name] = {
                            'precision': float(line_parts[1]),
                            'recall': float(line_parts[2]),
                            'f1-score': float(line_parts[3]),
                            'support': int(line_parts[4]) if len(line_parts) > 4 else 0
                        }
                    except (ValueError, IndexError):
                        continue 

        metrics_data = {
            'experiment_id': os.path.basename(self.experiment_dir),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_parameters': {
                'embedding_model': 'TF-IDF', # Se usa TF-IDF
                'classifier': 'SVM',
                'kernel': self.svm_kernel_parameter,
                'C': self.svm_c_parameter,
                'Gamma': self.svm_gamma_parameter,
            },
            'preprocessing_parameters': {
                'stemmer': 'SnowballStemmer-spanish',
                'remove_stopwords': True
            },
            'performance_metrics': {
                'cross_validation_scores': {
                    'mean': float(np.mean(results['cv_scores'])),
                    'std': float(np.std(results['cv_scores'])),
                    'scores': results['cv_scores'].tolist()
                },
                'classification_metrics': classification_dict,
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'recall': results['recall'],
                'precision': results['precision']
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
        'tfidf_vectorizer': TfidfVectorizer() # Guardar el vectorizador TF-IDF
    }
    with open(os.path.join(analyzer.experiment_dir, f'sentiment_model_{count}.pkl'), 'wb') as f:
        pickle.dump(components,f)
    
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
