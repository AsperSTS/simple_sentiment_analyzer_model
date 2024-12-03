import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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

nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.label_encoder = LabelEncoder()
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words = set(stopwords.words('spanish'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = SVC(kernel='rbf', probability=True)


    def preprocess_text(self, text):
        """Preprocesa el texto aplicando normalización básica."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def get_bert_embedding(self, text):
        """Obtiene embeddings usando BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def prepare_data(self, df):
        """Prepara datos y genera embeddings TF-IDF y BERT."""
        sentiment_mapping = {
            1: 'alegria', 6: 'alegria',
            2: 'tristeza',
            3: 'estres', 9: 'estres',
            4: 'inquietud', 5: 'inquietud',
            7: 'miedo', 10: 'miedo',
            8: 'enojo'
        }

        X_text = []
        y = []
        
        for idx, row in df.iterrows():
            for q_num in range(1, 11):
                col_name = f"{q_num}. "
                cols = [col for col in df.columns if col.startswith(col_name)]
                if cols:
                    text = row[cols[0]]
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        X_text.append(processed_text)
                        y.append(sentiment_mapping[q_num])

        X_text = np.array(X_text)
        y = self.label_encoder.fit_transform(y)
        
        # Generar TF-IDF y embeddings BERT
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text).toarray()
        X_bert = np.array([self.get_bert_embedding(text)[0] for text in X_text])
        
        return X_tfidf, X_bert, y

    def train_model(self, X, y, algorithm='svm'):
        """Entrena y evalúa el modelo especificado."""
        if algorithm == 'svm':
            model = SVC(kernel='rbf', probability=True)
        elif algorithm == 'naive_bayes':
            model = MultinomialNB()
        elif algorithm == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError("Algoritmo no soportado: elige entre 'svm', 'naive_bayes', o 'knn'.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X, y, cv=5)

        y_pred = model.predict(X_test)
        return {
            'model': model,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred, target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test, y_pred)
        }

    def plot_results(self, results, title="Resultados del Modelo"):
        """Visualiza resultados del modelo."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                    xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title(f'Matriz de Confusión - {title}')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_scores'])
        plt.title(f'Distribución de Validación Cruzada - {title}')
        plt.ylabel('Puntuación')
        plt.show()

    # def predict_sentiment(self, text, model):
    #     """Predice el sentimiento usando un modelo dado."""
    #     processed_text = self.preprocess_text(text)
    #     embedding = self.get_bert_embedding(processed_text)
    #     prediction = model.predict(embedding)
    #     probabilities = model.predict_proba(embedding)
    #     sentiment = self.label_encoder.inverse_transform(prediction)[0]
    #     return sentiment, dict(zip(self.label_encoder.classes_, probabilities[0]))
    def predict_sentiment(self, text):
        """Predice el sentimiento para un nuevo texto."""
        # Preprocesar el texto
        processed_text = self.preprocess_text(text)
        
        # Generar embedding usando BERT
        embedding = self.get_bert_embedding(processed_text)
        
        # Realizar predicción
        prediction = self.classifier.predict(embedding)
        probabilities = self.classifier.predict_proba(embedding)
        
        # Decodificar la predicción a etiqueta de clase
        sentiment = self.label_encoder.inverse_transform(prediction)[0]
        
        # Crear un diccionario de probabilidades por clase
        prob_dict = dict(zip(self.label_encoder.classes_, probabilities[0]))
        
        return sentiment, prob_dict

# def main():
#     df = pd.read_csv('Textos_Dataset_Completo_utf8.csv')
#     analyzer = SentimentAnalyzer()

#     X_tfidf, X_bert, y = analyzer.prepare_data(df)
    
#     # Entrenar y evaluar modelos con TF-IDF
#     print("\nEntrenando modelo con TF-IDF...")
#     results_tfidf_svm = analyzer.train_model(X_tfidf, y, algorithm='svm')
#     analyzer.plot_results(results_tfidf_svm, title="SVM con TF-IDF")
    
#     print("\nEntrenando modelo con Naive Bayes...")
#     results_tfidf_nb = analyzer.train_model(X_tfidf, y, algorithm='naive_bayes')
#     analyzer.plot_results(results_tfidf_nb, title="Naive Bayes con TF-IDF")

#     # Entrenar y evaluar modelos con BERT
#     print("\nEntrenando modelo con embeddings BERT...")
#     results_bert_svm = analyzer.train_model(X_bert, y, algorithm='svm')
#     analyzer.plot_results(results_bert_svm, title="SVM con BERT")

# if __name__ == "__main__":
#     main()
# Inicializa el analizador de sentimientos
analyzer = SentimentAnalyzer()

# Carga los datos y entrena el modelo (asegúrate de completar estos pasos previamente)
df = pd.read_csv('Textos_Dataset_Completo_utf8.csv')
X_tfidf, X_bert, y = analyzer.prepare_data(df)
analyzer.train_model(X_tfidf, y)

# Realiza una predicción
texto = "Estoy muy preocupado por el examen de mañana"
sentimiento, probabilidades = analyzer.predict_sentiment(texto)

# Muestra los resultados
print(f"Texto: {texto}")
print(f"Sentimiento predicho: {sentimiento}")
print("Probabilidades por clase:")
for clase, prob in probabilidades.items():
    print(f"{clase}: {prob:.2f}")
