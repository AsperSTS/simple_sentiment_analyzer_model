import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.corpus import stopwords
import re
import nltk
import warnings
from imblearn.over_sampling import SMOTE
from spacy import load
import time 
from utils import AnalyzerUtils
import os
warnings.filterwarnings('ignore')


class SentimentAnalyzer:

    def __init__(self, imported_class):
        """
        Constructor de la clase SentimentAnalyzer.
        
        Args:
            imported_class (bool): True si se va a importar una clase desde otro archivo, False en caso contrario.
        """
        self.imported_class = imported_class
        self.utils = AnalyzerUtils(self)
        
        self.generate_train_test_data = True
        self.remarks = "None"    
        self.pretrained_model_name = "PlanTL-GOB-ES/roberta-base-bne"
                    
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.label_encoder = LabelEncoder()
        
        # {'C': 7.213419527486499, 'class_weight': 'balanced', 'degree': 2, 'gamma': 0.6069599747810114, 'kernel': 'rbf'}
        # Initialize SVM parameters
        self.svm_c_parameter = 7.213419527486499 #1 #9.795846
        self.svm_kernel_parameter = 'rbf' #'rbf'
        self.svm_gamma_parameter = 0.6069 #'scale' #0.39615023
        self.svm_tolerance_parameter = 0.001
        self.svm_class_weight_parameter = None
        
        
        # Initialize all classifiers
        self.svm_classifier = SVC(kernel=self.svm_kernel_parameter, probability=True, 
                            C=self.svm_c_parameter, tol=self.svm_tolerance_parameter, 
                            class_weight=self.svm_class_weight_parameter, gamma=self.svm_gamma_parameter, degree=2)
        self.svm_precision_result = None
        
        self.nb_classifier = GaussianNB()
        self.knn_classifier = KNeighborsClassifier(n_neighbors=6)
        
        # Definir espacio de búsqueda para RandomizedSearchCV
        self.param_distributions = {
            'C': uniform(0.1, 10.0),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': uniform(0.001, 1.0),
            'class_weight': ['balanced', None],
            'degree': randint(2, 5)  # Solo para kernel poly
        }
        
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))
        if not self.imported_class:
            self.experiment_dir, self.current_run_number = self.utils.create_experiments_directory("experiments")
            self.models_dir = self.utils.create_models_directory("best_models")

    def preprocess_text(self, text):
        """
        Preprocesa el texto aplicando normalización básica y lematización.
        
        Parameters:
        text (str): El texto a preprocesar.
        
        Returns:
        str: El texto preprocesado.
        """
        print("Preprocesando el texto...")
        
        # Verificar que se pase un string
        if not isinstance(text, str):
            return ""      
        
        # Eliminar puntuación y caracteres especiales excepto acentos y ñ
        text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
        
        # Normalizar a minúsculas
        text = text.lower()
        
        # Eliminar números si no son relevantes (no se hace por defecto)
        text = re.sub(r'\d+', '', text)
        
        # Manejar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        
        # Lematización en lugar de stemming para mantener mejor el significado
        # Se utiliza el modelo de lenguaje español de spaCy
        
        # Cargar el modelo de lenguaje español
        nlp = load('es_core_news_sm')
        
        # Procesar el texto con el modelo de lenguaje
        doc = nlp(text)
        
        # Extraer los tokens lematizados sin stopwords y con longitud mayor a 2
        tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 2]
        
        return ' '.join(tokens)
    def get_bert_embedding_cls_mean(self, text):
        """
        Obtiene el embedding BERT para un texto dado, utilizando el token [CLS] y la media de todos los embeddings.
        
        Parameters:
        text (str): El texto para el que se obtendrá el embedding BERT.
        
        Returns:
        numpy.ndarray: El embedding BERT para el texto, con shape (1, 768).
        """
        print("Obteniendo embedding BERT...")
        
        # Tokenización
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            # Obtener las salidas del modelo
            outputs = self.model(**inputs)
            
        # Usar el token [CLS] (primera posición) para la representación global del texto
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Usar la media de todos los embeddings (promedio de todos los tokens)
        mean_embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Combinar [CLS] y la media
        combined_embedding = torch.cat((cls_embedding, mean_embedding), dim=1)
        
        return combined_embedding.numpy()
    
    ## VERSIÓN ANTERIOR
    def prepare_data(self, df):
        """
        Prepara los datos para el entrenamiento.

        Toma un DataFrame como parámetro y devuelve dos arrays numpy: X con las
        características de los textos y y con las etiquetas correspondientes.

        Primero, se mapean las preguntas a sentimientos utilizando un diccionario
        sentiment_mapping. Luego, se itera sobre cada fila del DataFrame y se
        para normalizar y preprocesar cada texto. Si el texto procesado no es vacío,
        se obtiene su embedding BERT y se agrega a la lista X, junto con su etiqueta
        correspondiente en la lista y.

        Finalmente, se utiliza la técnica de oversampling SMOTE para balancear las
        clases y se devuelve el par (X_balanced, y_balanced) con los datos balanceados.
        """
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
        
        # Mapeo de números de pregunta a nombres de columna
        column_mapping = {i: f"pregunta_{i}" for i in range(1, 11)}
        
        for idx, row in df.iterrows():
            for q_num, column_name in column_mapping.items():
                if column_name in df.columns:
                    text = row[column_name]
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        embedding = self.get_bert_embedding_cls_mean(processed_text)
                        X.append(embedding[0])
                        y.append(sentiment_mapping[q_num])
        
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        return X_balanced, y_balanced
    


    def find_best_parameters(self, X, y):
        """Versión mejorada de train_svm con RandomizedSearchCV."""
        print("Entrenando modelo con RandomizedSearchCV...")
        
        # División del dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Configurar y ejecutar RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.svm_classifier,
            param_distributions=self.param_distributions,
            n_iter=25,  # Número de combinaciones a probar
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1  # Usar todos los cores disponibles
        )
        
        # Entrenar el modelo
        random_search.fit(X_train, y_train)
        
        # Actualizar el clasificador con los mejores parámetros
        self.svm_classifier = random_search.best_estimator_
        
        # Evaluación en conjunto de prueba
        y_pred = self.svm_classifier.predict(X_test)
        
        # Retornar resultados
        return {
            'cv_scores': random_search.cv_results_['mean_test_score'],
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred)
        }
    def train_svm(self, X, y):  
        print("Entrenando modelo svm...")
        """Entrena el modelo de clasificación."""
        # División del dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenamiento del modelo
        self.svm_classifier.fit(X_train, y_train)
        
        # Validación cruzada
        cv_scores = cross_val_score(self.svm_classifier, X, y, cv=5)
        
        # Evaluación en conjunto de prueba
        y_pred = self.svm_classifier.predict(X_test)
        
        return {
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred)
        }
    def train_naive_bayes(self, X, y):
        print("Entrenando modelo naive bayes...")
        """Entrena y evalúa un modelo Naive Bayes."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.nb_classifier.fit(X_train, y_train)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(self.nb_classifier, X, y, cv=skf)
        # Validación cruzada
        # cv_scores = cross_val_score(self.nb_classifier, X, y, cv=5)
        
        # Evaluación
        y_pred = self.nb_classifier.predict(X_test)
        
        return {
            'model': self.nb_classifier,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred)
        }

    def train_knn(self, X, y):
        print("Entrenando modelo knn...")
        """Entrena y evalúa un modelo KNN."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.knn_classifier.fit(X_train, y_train)
        
        # Validación cruzada
        cv_scores = cross_val_score(self.knn_classifier, X, y, cv=5)
        
        # Evaluación
        y_pred = self.knn_classifier.predict(X_test)
        
        return {
            'model': self.knn_classifier,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred)
        }
        
    ## VERSIÓN ANTERIOR
    def predict_sentiment(self, text):
        print("Prediciendo sentimiento...")
        """Predice el sentimiento para un nuevo texto."""
        processed_text = self.preprocess_text(text)
        embedding = self.get_bert_embedding_cls_mean(processed_text)
        prediction = self.svm_classifier.predict(embedding)
        probabilities = self.svm_classifier.predict_proba(embedding)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))
    

def main():
    
    # Cargar datos
    # df = pd.read_csv('dataset_normalizado_utf8.csv')
    
    """
    Main function to execute the sentiment analysis pipeline.

    This function performs the following tasks:
    1. Loads the dataset from a specified CSV file.
    2. Initializes the SentimentAnalyzer instance.
    3. Collects user input for observations and dataset preparation preferences.
    4. Conducts exploratory data analysis (EDA) and saves the results.
    5. Prepares or loads training and test data based on user input.
    6. Trains and evaluates multiple classifiers: SVM, Naive Bayes, and KNN.
    7. Saves model metrics and visualizes results.
    8. Saves the trained SVM model components for future use.
    9. Predicts sentiment for a list of sample texts and prints the results.
    """
    df = pd.read_csv('dataset_normalizado_utf8_aumentado.csv')
    # 
    df = df[df['edad'] <= 30]
    df = df[df['grado_estudios'] != "Maestría"]
    df = df[df['nivel_socioeconomico'] != "Alto"] # Con esta solamente, sale chido
    
    analyzer = SentimentAnalyzer(False)
    print(analyzer.experiment_dir)
    
    # Input para las observaciones, con color amarillo para resaltar
    analyzer.remarks = input(analyzer.utils.color_texto("Ingresa tus modificaciones o observaciones: ", 'amarillo'))
    
    # Input para preparar un nuevo train dataset o cargar desde archivos npy
    respuesta = input(
        """
        ¿Quieres preparar un nuevo train dataset? 
        
        """
        + analyzer.utils.color_texto("Esto tomará aproximadamente 800 segundos.", "rojo")
        + """
        
        (y/n): 
        """
    )
    analyzer.generate_train_test_data = (lambda x: x.lower() in ('y', ''))(respuesta)
    
    # Ejecutar el análisis
    results_eda = analyzer.utils.perform_eda(df)
    analyzer.utils.save_eda(results_eda)
    
    # Manejar datos de entrenamiento y prueba
    try:
        if analyzer.generate_train_test_data:
            # Generar datos de entrenamiento desde cero
            X, y = analyzer.prepare_data(df)
            # Guardar datos para uso futuro
            analyzer.utils.save_train_test_data(X, y, file_prefix="prepared_data")
        else:
            # Cargar datos previamente guardados
            X, y, analyzer.label_encoder = analyzer.utils.load_train_test_data(file_prefix="prepared_data")
    except FileNotFoundError as e:
        print(f"Error: {e}. Generando datos desde cero.")
        X, y = analyzer.prepare_data(df)
        analyzer.utils.save_train_test_data(X, y, file_prefix="prepared_data")
        
    # # Ejecutar el análisis para buscar mejores parametros para SVM
    # best_results_svm = analyzer.find_best_parameters(X, y)    
    # print(best_results_svm['best_params'])
    
    
    if 'best_results_svm' not in locals() and 'best_results_svm' not in globals():
        start_time = time.time()
        results_svm = analyzer.train_svm(X, y)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución SVM: {execution_time:.2f} segundos")
        analyzer.utils.save_multi_model_metrics(results_svm, 'svm', execution_time)
        analyzer.utils.plot_results(results_svm, "svm")
        analyzer.utils.save_svm_experiment_metrics(results_svm, execution_time)
        
        
        start_time = time.time()
        results_nb = analyzer.train_naive_bayes(X, y)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución Naive Bayes: {execution_time:.2f} segundos")
        analyzer.utils.save_multi_model_metrics(results_nb, 'naive_bayes', execution_time)
        analyzer.utils.plot_results(results_nb, "naive_bayes")
        
        
        start_time = time.time()
        results_knn = analyzer.train_knn(X, y)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución KNN: {execution_time:.2f} segundos")
        analyzer.utils.save_multi_model_metrics(results_knn, 'knn', execution_time)
        analyzer.utils.plot_results(results_knn, "knn")
        
        # Lista de textos para testear el modelo
        texts = [
            "Me siento muy contento por haber logrado mis metas",
            "Estoy muy triste, no sé qué hacer",
            "Hoy es un buen día, estoy emocionado",
            "No me siento bien, creo que estoy enfermo",
            "Estoy muy enojado por lo que sucedió",
            "Estoy muy emocionado de empezar mi nuevo trabajo"
        ]
        
        # Saving SVM Model
        components_svm_model = {
            'classifier': analyzer.svm_classifier,
            'label_encoder': analyzer.label_encoder,
            'tokenizer': analyzer.tokenizer,
            'model': analyzer.model 
        }
        
        
        analyzer.utils.save_model(components_svm_model)
        
        # Recorrer la lista de textos para obtener las predicciones
        for text in texts:
            sentiment, probs = analyzer.predict_sentiment(text)
            print(f"\nTexto de ejemplo: {text}")
            print(f"Sentimiento predicho: {sentiment}")
            print("\nProbabilidades por clase:")
            for sentiment_class, prob in probs.items():
                print(f"{sentiment_class}: {prob:.2f}")
    else:
        os.rmdir(analyzer.experiment_dir)

if __name__ == "__main__":
    main()
