import pickle
import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
import os

class SentimentPrecedictor:
    def __init__(self):
        self.results_path = "experiments"
        
    def get_available_models(self):
        """
        Obtiene una lista de los modelos disponibles en el directorio 'results'.

        Returns:
            dict: Diccionario con los nombres de los modelos y sus rutas.
        """
        
        # Encontrar el siguiente número de ejecución
        existing_runs = [d for d in os.listdir(self.results_path) if d.startswith('run')]
        if not existing_runs:
            return 0
        else:
            run_numbers = [int(run.replace('run', '')) for run in existing_runs]
            next_run = max(run_numbers) + 1
            
            
        results_dir = self.results_path
        model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        return {d.name: d for d in model_dirs}

    def load_model(self):
        with open('sentiment_model_2.pkl', 'rb') as f:
            components = pickle.load(f)
        
        analyzer = SentimentAnalyzer()
        analyzer.classifier = components['classifier']
        analyzer.label_encoder = components['label_encoder']
        analyzer.tokenizer = components['tokenizer']
        analyzer.model = components['model']
        return analyzer

def main():
    st.title("Analizador de Sentimientos")
    predictor = SentimentPrecedictor()

    analyzer = predictor.load_model()
    
   
    text = st.text_area("Ingrese el texto a analizar:")
    
    if st.button("Analizar"):
        if text:
            sentiment, probs = analyzer.predict_sentiment(text)
            st.write(f"**Sentimiento detectado:** {sentiment}")   
            st.write("**Probabilidades por clase:**")
            for sentiment_class, prob in probs.items():
                st.progress(prob)
                st.write(f"{sentiment_class}: {prob:.2f}")

if __name__ == "__main__":
    main()