import pickle
import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
import os
from pathlib import Path

class SentimentPredictor:
    def __init__(self):
        self.results_path = Path("experiments")

    def get_available_models(self):
        """
        Obtiene una lista de los modelos disponibles en el directorio 'experiments'.
        Busca subdirectorios que comiencen con 'run' y dentro de ellos archivos 
        que contengan 'model' en el nombre y tengan extensión '.pkl'.
        """
        if not self.results_path.exists():
            return {}

        # Buscar subdirectorios que comiencen con 'run'
        existing_runs = [self.results_path / d for d in os.listdir(self.results_path) if d.startswith('run') and (self.results_path / d).is_dir()]

        # Buscar modelos dentro de los subdirectorios encontrados
        models = {}
        for run_dir in existing_runs:
            for f in os.listdir(run_dir):
                if "model" in f and f.endswith(".pkl"):
                    models[f"{run_dir.name}/{f}"] = run_dir / f

        return models

    def load_model(self, model_path):
        """
        Carga el modelo guardado y configura el analizador.
        """
        try:
            with open(model_path, 'rb') as f:
                components = pickle.load(f)
            
            analyzer = SentimentAnalyzer(imported_class=True)
            
            # Asignar los componentes cargados al analizador
            analyzer.svm_classifier = components['classifier']
            analyzer.label_encoder = components['label_encoder']
            analyzer.tokenizer = components['tokenizer']
            analyzer.model = components['model']
            
            return analyzer
            
        except FileNotFoundError:
            st.error(f"No se encontró el archivo del modelo en {model_path}")
            return None
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None

def main():
    st.title("Analizador de Sentimientos")
    
    predictor = SentimentPredictor()
    available_models = predictor.get_available_models()
    
    if not available_models:
        st.error("No se encontraron modelos disponibles en el directorio 'experiments'.")
        return

    # Crear un menú desplegable para seleccionar el modelo
    model_name = st.selectbox("Seleccione un modelo:", list(available_models.keys()))
    model_path = available_models[model_name]

    # Cargar el modelo seleccionado
    analyzer = predictor.load_model(model_path)
    
    if analyzer is None:
        st.error("No se pudo cargar el modelo seleccionado.")
        return
    
    # Crear el área de texto para el input
    text = st.text_area("Ingrese el texto a analizar:", height=100)
    
    if st.button("Analizar"):
        if not text:
            st.warning("Por favor ingrese un texto para analizar.")
            return
            
        try:
            # Realizar la predicción
            sentiment, probs = analyzer.predict_sentiment(text)
            
            # Mostrar resultados
            st.success(f"Sentimiento detectado: {sentiment}")
            
            # st.write("Probabilidades por clase:")
            # for sentiment_class, prob in probs.items():
            #     st.write(f"{sentiment_class}:")
            #     st.progress(float(prob))
            #     st.write(f"{prob:.2f}")
                
        except Exception as e:
            st.error(f"Error al analizar el texto: {str(e)}")
            
if __name__ == "__main__":
    main()
