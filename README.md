# Analizador de Sentimientos con SVM y Embeddings de Roberta

Este proyecto implementa un analizador de sentimientos en español que utiliza:

* **SVM (Support Vector Machine)** como algoritmo principal de clasificación.
* **Embeddings de Roberta (PlanTL-GOB-ES/roberta-base-bne)** para la representación vectorial de las palabras.

## Características

* Preprocesamiento de texto: normalización, tokenización, eliminación de stop words y lematización.
* Extracción de embeddings de Roberta.
* Entrenamiento de un modelo SVM con kernel RBF.
* Evaluación del modelo con validación cruzada y métricas de clasificación.
* Visualización de resultados con matrices de confusión y boxplots.
* Guardado del modelo entrenado en un archivo pickle.
* Predicción de sentimientos para nuevos textos.

## Requisitos

* Python 3.7 o superior
* Las bibliotecas listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/AsperSTS/simple_sentiment_analyzer_model.git