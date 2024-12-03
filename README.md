# Analizador de Sentimientos con SVM y Embeddings de Roberta

Un analizador de sentimientos en español que combina Support Vector Machine (SVM) con embeddings del modelo Roberta para clasificación de texto.

## Descripción

Este proyecto implementa un sistema de análisis de sentimientos utilizando:

- SVM (Support Vector Machine) como algoritmo de clasificación
- Embeddings contextuales del modelo PlanTL-GOB-ES/roberta-base-bne
- Técnicas avanzadas de procesamiento de lenguaje natural

## Características Principales

### Procesamiento de Texto
- Normalización y limpieza de texto
- Tokenización especializada para español
- Eliminación de stop words
- Lematización mediante spaCy

### Modelo y Análisis
- Extracción de embeddings usando Roberta
- Clasificación mediante SVM con kernel RBF
- Evaluación completa del modelo mediante:
  - Validación cruzada
  - Métricas de clasificación
  - Matrices de confusión
  - Visualización mediante boxplots

### Funcionalidades
- Entrenamiento automatizado del modelo
- Guardado y carga del modelo entrenado
- Predicción de sentimientos para textos nuevos
- Visualización de resultados y métricas

## Requisitos del Sistema

- Python 3.7 o superior
- Dependencias especificadas en `requirements.txt`
- Espacio en disco suficiente para modelos y embeddings

## Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/AsperSTS/simple_sentiment_analyzer_model.git
cd simple_sentiment_analyzer_model
```

### 2. Configurar el Entorno

#### Opción A: Usando Anaconda (Recomendado), incluye dependencias
```bash
# Crear el entorno desde el archivo environment.yml
conda env create -f environment.yml

# Activar el entorno
conda activate sentiment_analyzer_env
```

#### Opción B: Usando pip
```bash
# Crear un entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Instalar Recursos Lingüísticos
```bash
python -m spacy download es_core_news_sm
```

## Notas Importantes

- En caso de errores durante la instalación, revisar `environment.yml` o `requirements.txt`
- Si hay problemas con la descarga de stopwords de spaCy, se puede eliminar esa línea de los archivos de dependencias
- Asegurarse de tener suficiente espacio en disco para los modelos de embeddings

## Contacto y Soporte

Para reportar problemas o sugerir mejoras, por favor crear un issue en el repositorio de GitHub.