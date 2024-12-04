import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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
warnings.filterwarnings('ignore')

class SentimentAnalyzer:

    def __init__(self):
            self.utils = AnalyzerUtils(self)
            
            self.generate_train_test_data = False
            
            self.pretrained_model_name = "PlanTL-GOB-ES/roberta-base-bne"
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            self.model = AutoModel.from_pretrained(self.pretrained_model_name)
            self.label_encoder = LabelEncoder()
            
            # Initialize SVM parameters
            self.svm_c_parameter = 1
            self.svm_kernel_parameter = 'linear'
            self.svm_tolerance_parameter = 0.01
            self.svm_class_weight_parameter = None
            
            # Initialize all classifiers
            self.svm_classifier = SVC(kernel=self.svm_kernel_parameter, probability=True, 
                                C=self.svm_c_parameter, tol=self.svm_tolerance_parameter, 
                                class_weight=self.svm_class_weight_parameter)
            self.nb_classifier = GaussianNB()
            self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
            
            
            nltk.download('punkt')
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('spanish'))
            self.experiment_dir = self.utils.create_experiment_directory()

    def preprocess_text(self, text):
        print("Preprocesando el texto...")
        """Preprocesa el texto aplicando normalización básica."""
        if not isinstance(text, str):
            return ""     
        
        text = re.sub(r'[^\w\sáéíóúñü]', '', text)  # Mantener acentos y ñ
        text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
        # Lematización en lugar de stemming para mantener mejor el significado
        nlp = load('es_core_news_sm')
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        
        return ' '.join(tokens)

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
        
        # Validación cruzada
        cv_scores = cross_val_score(self.nb_classifier, X, y, cv=5)
        
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
    def predict_sentiment(self, text):
        print("Prediciendo sentimiento...")
        """Predice el sentimiento para un nuevo texto."""
        processed_text = self.preprocess_text(text)
        embedding = self.get_bert_embedding(processed_text)
        prediction = self.svm_classifier.predict(embedding)
        probabilities = self.svm_classifier.predict_proba(embedding)
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))
 
def main():
    start_time = time.time()
    # Cargar datos
    df = pd.read_csv('Textos_Dataset_Completo_utf8.csv')
    analyzer = SentimentAnalyzer()
    
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
