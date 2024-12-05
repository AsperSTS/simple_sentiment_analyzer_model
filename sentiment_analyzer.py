import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.corpus import stopwords
import re
import nltk
import warnings
from imblearn.over_sampling import SMOTE,ADASYN
from spacy import load
import time 
from utils import AnalyzerUtils
warnings.filterwarnings('ignore')


class SentimentAnalyzer:

    def __init__(self):
            self.utils = AnalyzerUtils(self)
            
            # Añadir nuevos componentes
            self.ngram_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                min_df=2
            )
            
            
            self.generate_train_test_data = True
            self.remarks = "None"
            
            self.pretrained_model_name = "PlanTL-GOB-ES/roberta-base-bne"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            self.model = AutoModel.from_pretrained(self.pretrained_model_name)
            self.label_encoder = LabelEncoder()
            
            # Initialize SVM parameters
            self.svm_c_parameter = 9.795846
            self.svm_kernel_parameter = 'rbf'
            self.svm_gamma_parameter = 0.39615023
            self.svm_tolerance_parameter = 0.001
            self.svm_class_weight_parameter = None
            
            # Initialize all classifiers
            self.svm_classifier = SVC(kernel=self.svm_kernel_parameter, probability=True, 
                                C=self.svm_c_parameter, tol=self.svm_tolerance_parameter, 
                                class_weight=self.svm_class_weight_parameter, gamma=self.svm_gamma_parameter)
            self.nb_classifier = ComplementNB(alpha=0.1)#GaussianNB()
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
            self.experiment_dir = self.utils.create_experiment_directory()

    def preprocess_text(self, text):
        print("Preprocesando el texto...")
        """Preprocesa el texto aplicando normalización básica."""
        if not isinstance(text, str):
            return ""     
        
        # text = re.sub(r'[^\w\sáéíóúñü]', '', text)  # Mantener acentos y ñ
        # text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
        text = text.lower()  # Normalizar a minúsculas
        text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
        # Eliminar números si no son relevantes
        text = re.sub(r'\d+', '', text)
        # Manejar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        
        # Lematización en lugar de stemming para mantener mejor el significado
        nlp = load('es_core_news_sm')
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc 
          if not token.is_stop and len(token.lemma_) > 2]
        
        return ' '.join(tokens)

    # def get_bert_embedding(self, text):
        
    #     print("Obteniendo embedding BERT...")
    #     inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     # Usar la media de todos los embeddings
    #     return outputs.last_hidden_state.mean(dim=1).numpy()
    def get_bert_embedding(self, text):
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
    def prepare_data_2(self, df):
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
            1: "pregunta_1",
            2: "pregunta_2",
            3: "pregunta_3",
            4: "pregunta_4",
            5: "pregunta_5",
            6: "pregunta_6",
            7: "pregunta_7",
            8: "pregunta_8",
            9: "pregunta_9",
            10: "pregunta_10"
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
    
    def prepare_data(self, df):
        """Versión mejorada de prepare_data con manejo robusto de balance de clases."""
        print(f"Preparando datos para el entrenamiento...")
        
        # Mantener el mapping existente
        sentiment_mapping = {
            1: 'alegria', 6: 'alegria',
            2: 'tristeza',
            3: 'estres', 9: 'estres',
            4: 'inquietud', 5: 'inquietud',
            7: 'miedo', 10: 'miedo',
            8: 'enojo'
        }
        
        texts = []
        labels = []
        
        # Mapeo de números de pregunta a nombres de columna
        column_mapping = {i: f"pregunta_{i}" for i in range(1, 11)}
        
        # Recolectar textos y etiquetas
        for idx, row in df.iterrows():
            for q_num, column_name in column_mapping.items():
                if column_name in df.columns:
                    text = row[column_name]
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        texts.append(processed_text)
                        labels.append(sentiment_mapping[q_num])
        
        # Obtener características BERT
        bert_features = np.array([
            self.get_bert_embedding(text)[0] 
            for text in texts
        ])
        
        # Obtener características n-grama
        ngram_features = self.ngram_vectorizer.fit_transform(texts)
        
        # Combinar características
        X = np.hstack([
            bert_features,
            ngram_features.toarray()
        ])
        
        # Codificar etiquetas
        y = self.label_encoder.fit_transform(labels)
        
        # Verificar el balance de clases
        unique_labels, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        print("Distribución de clases original:", class_distribution)
        
        try:
            # Intentar ADASYN primero con parámetros ajustados
            adasyn = ADASYN(
                random_state=42,
                sampling_strategy='auto',
                n_neighbors=min(5, min(counts) - 1),  # Ajustar vecinos basado en el tamaño de la clase más pequeña
            )
            X_balanced, y_balanced = adasyn.fit_resample(X, y)
            print("Balanceo exitoso con ADASYN")
            
        except ValueError as e:
            print(f"ADASYN falló: {str(e)}")
            print("Intentando SMOTE como alternativa...")
            
            try:
                # Intentar SMOTE como respaldo
                smote = SMOTE(
                    random_state=42,
                    sampling_strategy='auto',
                    k_neighbors=min(5, min(counts) - 1)
                )
                X_balanced, y_balanced = smote.fit_resample(X, y)
                print("Balanceo exitoso con SMOTE")
                
            except ValueError as e:
                print(f"SMOTE también falló: {str(e)}")
                print("Usando datos originales sin balanceo...")
                X_balanced, y_balanced = X, y
        
        # Verificar el balance final
        unique_labels, counts = np.unique(y_balanced, return_counts=True)
        final_distribution = dict(zip(unique_labels, counts))
        print("Distribución de clases final:", final_distribution)
        
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
            n_iter=100,  # Número de combinaciones a probar
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
        print("Entrenando modelo...")
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
    def predict_sentiment_2(self, text):
        print("Prediciendo sentimiento...")
        """Predice el sentimiento para un nuevo texto."""
        processed_text = self.preprocess_text(text)
        embedding = self.get_bert_embedding(processed_text)
        prediction = self.svm_classifier.predict(embedding)
        probabilities = self.svm_classifier.predict_proba(embedding)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))
    
    def predict_sentiment(self, text):
        """Versión actualizada de predict_sentiment para trabajar con las nuevas características."""
        print("Prediciendo sentimiento...")
        processed_text = self.preprocess_text(text)
        
        # Obtener embedding BERT
        bert_embedding = self.get_bert_embedding(processed_text)
        
        # Obtener características n-grama
        ngram_features = self.ngram_vectorizer.transform([processed_text])
        
        # Combinar características
        combined_features = np.hstack([
            bert_embedding,
            ngram_features.toarray()
        ])
        
        # Realizar predicción
        prediction = self.svm_classifier.predict(combined_features)
        probabilities = self.svm_classifier.predict_proba(combined_features)
        
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))
    def color_texto(self, texto, color):
        colores = {
            'rojo': '\033[91m',
            'verde': '\033[92m',
            'azul': '\033[94m',
            'amarillo': '\033[93m',
            'fin': '\033[0m',  # Restablece el color
        }
        return colores.get(color, '') + texto + colores['fin']
def main():
    start_time = time.time()
    # Cargar datos
    df = pd.read_csv('dataset_normalizado_utf8.csv')
    
    # df = df[df['edad'] <= 30]
    # df = df[df['grado_estudios'] != "Maestría"]
    
    df = df[df['nivel_socioeconomico'] != "Alto"] # COn esta solamente, sale chido
    
    analyzer = SentimentAnalyzer()
    
    # Input para las observaciones, con color amarillo para resaltar
    analyzer.remarks = input(analyzer.color_texto("Ingresa tus modificaciones o observaciones: ", 'amarillo'))
    

    # Input para preparar un nuevo train dataset o cargar desde archivos npy
    respuesta = input(
        """
        ¿Quieres preparar un nuevo train dataset? 
        
        """
        + analyzer.color_texto("Esto tomará aproximadamente 800 segundos.", "rojo")
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
        
        
    results_svm = analyzer.train_svm(X, y)
    results_nb = analyzer.train_naive_bayes(X, y)
    results_knn = analyzer.train_knn(X, y)
    
    # print(results_svm['best_params'])
    # print(results_svm['best_score'])
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    analyzer.utils.save_multi_model_metrics(results_svm, 'svm', execution_time)
    analyzer.utils.save_multi_model_metrics(results_nb, 'naive_bayes', execution_time)
    analyzer.utils.save_multi_model_metrics(results_knn, 'knn', execution_time)
    
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

    # Guardar métricas del experimento
    analyzer.utils.save_svm_experiment_metrics(results_svm, execution_time)
    
    # Visualizar resultados
    analyzer.utils.plot_results(results_svm, "svm")
    analyzer.utils.plot_results(results_knn, "knn")
    analyzer.utils.plot_results(results_nb, "naive_bayes")
    
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
    components = {
        'classifier': analyzer.svm_classifier,
        'label_encoder': analyzer.label_encoder,
        'tokenizer': analyzer.tokenizer,
        'model': analyzer.model
    }
    analyzer.utils.save_model(components)
    
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
