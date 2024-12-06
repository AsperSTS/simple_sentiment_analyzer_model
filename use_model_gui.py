import pickle
import streamlit as st
import os
from pathlib import Path
from sentiment_analyzer import SentimentAnalyzer
import json
from PIL import Image

class SentimentPredictor:
    def __init__(self):
        self.results_path = Path("experiments")
        self.model_path = None
    def get_available_models(self):
        if not self.results_path.exists():
            return {}

        existing_runs = [self.results_path / d for d in os.listdir(self.results_path) if d.startswith('run') and (self.results_path / d).is_dir()]
        models = {}
        for run_dir in existing_runs:
            for f in os.listdir(run_dir):
                if "model" in f and f.endswith(".pkl"):
                    models[f"{run_dir.name}/{f}"] = run_dir / f
        return models

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                components = pickle.load(f)

            analyzer = SentimentAnalyzer(imported_class=True)
            analyzer.svm_classifier = components['classifier']
            analyzer.label_encoder = components['label_encoder']
            analyzer.tokenizer = components['tokenizer']
            analyzer.model = components['model']
            return analyzer
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None
@st.cache_data
def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error al cargar el archivo JSON: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    
    st.title("Analizador de Sentimientos con SVM y Embeddings de Roberta")
    
    # Inicializar predictor y modelos
    predictor = SentimentPredictor()
    available_models = predictor.get_available_models()
    
    if not available_models:
        st.error("No se encontraron modelos disponibles en el directorio 'experiments'.")
        return

    # Cargar JSON y visualizaciones
    # experiment_metrics = load_json_data("experiment_metrics.json")
    # eda_results = load_json_data("eda_results.json")

    # Crear pestañas
    tabs = st.tabs(["Selección de Modelo", "EDA", "Estimador", "Resultados del Experimento", "Comparacion de Algoritmos"])

    with tabs[0]:
        st.header("Seleccione un modelo")
        model_name = st.selectbox("Modelos disponibles:", list(available_models.keys()))
        predictor.model_path = available_models[model_name]
        with st.spinner('Cargando modelo...'):
            analyzer = predictor.load_model(predictor.model_path)
        # analyzer = predictor.load_model(predictor.model_path)
        print()
        if analyzer:
            st.success(f"Modelo {model_name} cargado exitosamente.")
            experiment_metrics = load_json_data(os.path.join(os.path.dirname(predictor.model_path), "experiment_metrics.json")) #if model_path else load_json_data("experiment_metrics.json")
            eda_results = load_json_data(os.path.join(os.path.dirname(predictor.model_path),"eda_results.json"))

    with tabs[1]:
        st.header("Resultados de EDA")
        if eda_results:
            st.subheader("Estadísticas Demográficas")
            st.write("Distribución de Género:")
            st.bar_chart(eda_results["demographics"]["genero_dist"])
            st.write("Distribución de Nivel Socioeconómico:")
            st.bar_chart(eda_results["demographics"]["nivel_socioeco_dist"])
            # st.write("Distribución Geográfica:")
            # st.map(eda_results["geographic_distribution"]["estado_dist"])
            
            st.image(os.path.join(os.path.dirname(predictor.model_path),"eda_visualizations.png"), caption="EDA Visualizations", use_container_width=True)

    with tabs[2]:
        st.header("Analisis de sentimientos")
        text = st.text_area("Ingrese el texto a analizar:", height=100)
        if st.button("Analizar"):
            if not text:
                st.warning("Por favor ingrese un texto.")
            else:
                sentiment, probs = analyzer.predict_sentiment(text)
                st.success(f"Sentimiento detectado: {sentiment}")
                st.write("Probabilidades por clase:")
                st.json(probs)

    with tabs[3]:
        st.header("Resultados del Experimento")
        if experiment_metrics:
            st.subheader("Parámetros del Modelo")
            st.json(experiment_metrics["model_parameters"])
            # st.subheader("Validación Cruzada:")
            # st.write(f"Media: {experiment_metrics['performance_metrics']['cross_validation_scores']['mean']:.4f}")
            # st.write(f"Desviación estándar: {experiment_metrics['performance_metrics']['cross_validation_scores']['std']:.4f}")
            st.subheader("Metricas")
            st.json(experiment_metrics["performance_metrics"])
            # st.image(os.path.join(os.path.dirname(predictor.model_path),"matriz_confusion_svm.png"), caption="Matriz de Confusión", use_container_width=True)
    with tabs[4]:
        st.header("Comparacion de Algoritmos")
        if experiment_metrics:
            # st.subheader("Resultados")
            # Crea tres columnas con el mismo ancho
            col1, col2, col3 = st.columns(3)

            # Muestra cada imagen en una columna
            with col1:
                st.image(os.path.join(os.path.dirname(predictor.model_path),"matriz_confusion_svm.png"), caption="Matriz de Confusión SVM", use_container_width=True)
            
            with col2:
                st.image(os.path.join(os.path.dirname(predictor.model_path),"matriz_confusion_knn.png"), caption="Matriz de Confusión KNN", use_container_width=True)

            with col3:
                st.image(os.path.join(os.path.dirname(predictor.model_path),"matriz_confusion_naive_bayes.png"), caption="Matriz de Confusión Naive Bayes", use_container_width=True)
            
if __name__ == "__main__":
    main()
