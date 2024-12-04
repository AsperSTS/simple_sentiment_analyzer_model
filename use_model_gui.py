import pickle
import streamlit as st
from sentiment_analyzer import SentimentAnalyzer


def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        components = pickle.load(f)
    
    analyzer = SentimentAnalyzer()
    analyzer.classifier = components['classifier']
    analyzer.label_encoder = components['label_encoder']
    analyzer.tokenizer = components['tokenizer']
    analyzer.model = components['model']
    return analyzer

def main():
    st.title("Analizador de Sentimientos")
    
    analyzer = load_model()
    
   
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