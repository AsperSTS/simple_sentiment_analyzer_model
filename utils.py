# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy import stats
import os
import pickle
import json
import joblib
import datetime
class AnalyzerUtils:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def create_experiment_directory(self):
        """
        Crea un nuevo directorio para el experimento actual.
        """
        base_dir = 'experiments'
        os.makedirs(base_dir, exist_ok=True)
        
        # Encontrar el siguiente número de ejecución
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run')]
        if not existing_runs:
            next_run = 1
        else:
            run_numbers = [int(run.replace('run', '')) for run in existing_runs]
            next_run = max(run_numbers) + 1
            
        # Crear directorio para esta ejecución
        run_dir = os.path.join(base_dir, f'run{next_run}')
        os.makedirs(run_dir)
        
        return run_dir
    def contar_pk1_mas_uno(self, directorio='.'):

        contador = 0
        for archivo in os.listdir(directorio):
            if archivo.endswith(".pkl"):
                contador += 1
        return contador + 1
    def save_model(self, components):
        with open(os.path.join(self.analyzer.experiment_dir,f'sentiment_model_.pkl'), 'wb') as f:
            pickle.dump(components, f)
        
    def save_svm_experiment_metrics(self, results, execution_time):
        """
        Guarda las métricas y parámetros del experimento en un archivo JSON.
        """
        print("Guardando métricas del experimento...")
        
        # Convertir el classification report de string a diccionario de una manera más robusta
        classification_dict = {}
        report_lines = results['classification_report'].split('\n')
        for line in report_lines:
            if line and not line.startswith('micro') and not line.startswith('macro') and not line.startswith('weighted') and not line.startswith('accuracy'):
                line_parts = line.strip().split()
                if len(line_parts) >= 4:  # Asegurarse de que la línea tiene suficientes elementos
                    class_name = line_parts[0]
                    try:
                        classification_dict[class_name] = {
                            'precision': float(line_parts[1]),
                            'recall': float(line_parts[2]),
                            'f1-score': float(line_parts[3]),
                            'support': int(line_parts[4]) if len(line_parts) > 4 else 0
                        }
                    except (ValueError, IndexError):
                        continue  # Saltarse líneas que no pueden ser parseadas correctamente

        metrics_data = {
            'experiment_id': os.path.basename(self.analyzer.experiment_dir),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_parameters': {
                'embedding_model': self.analyzer.pretrained_model_name,
                'classifier': 'SVM',
                'kernel': self.analyzer.svm_kernel_parameter,
                'C': self.analyzer.svm_c_parameter,
                'tolerance': self.analyzer.svm_tolerance_parameter,
                'class_weight': self.analyzer.svm_class_weight_parameter
            },
            'preprocessing_parameters': {
                'stemmer': 'SnowballStemmer-spanish',
                'remove_stopwords': True,
                'max_length': 512,
            },
            'performance_metrics': {
                'cross_validation_scores': {
                    'mean': float(np.mean(results['cv_scores'])),
                    'std': float(np.std(results['cv_scores'])),
                    'scores': results['cv_scores'].tolist()
                },
                'classification_metrics': classification_dict,
                'confusion_matrix': results['confusion_matrix'].tolist()
            },
            'execution_metrics': {
                'total_execution_time': execution_time,
            }
        }
        
        # Guardar métricas en archivo JSON
        metrics_file = os.path.join(self.analyzer.experiment_dir, 'experiment_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=4)
        
        print(f"Métricas guardadas en: {metrics_file}")
    def save_multi_model_metrics(self, model_results, model_name, execution_time):
        metrics_path = os.path.join(self.analyzer.experiment_dir, f'{model_name}_metrics.txt')
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Métricas del experimento - {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Resultados de Validación Cruzada:\n")
            cv_scores = model_results['cv_scores']
            f.write(f"Media: {cv_scores.mean():.4f}\n")
            f.write(f"Desviación estándar: {cv_scores.std():.4f}\n\n")
            
            f.write("Reporte de Clasificación:\n")
            f.write(str(model_results['classification_report']) + "\n\n")
            
            f.write(f"Tiempo de ejecución: {execution_time:.2f} segundos\n")
    def perform_eda(self, df):
  
        analysis_results = {}
        
        # Basic dataset info
        analysis_results['shape'] = df.shape
        analysis_results['missing_values'] = df.isnull().sum()
        
        # Demographic analysis
        demographics = {
            'edad_stats': df['Edad:'].describe(),
            'genero_dist': df['Género:'].value_counts(),
            'nivel_socioeco_dist': df['Nivel socioeconómico:'].value_counts(),
            'education_dist': df['Grado de estudios:'].value_counts()
        }
        analysis_results['demographics'] = demographics
        
        # Text responses analysis
        text_columns = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.'))]
        text_analysis = {}
        for col in text_columns:
            text_analysis[col] = {
                'response_length': df[col].str.len().describe(),
                'word_count': df[col].str.split().str.len().describe()
            }
        analysis_results['text_analysis'] = text_analysis
        
        geo_dist = {
            'estado_dist': df['Estado de origen:'].value_counts(),
            'municipio_dist': df['Municipio de origen:'].value_counts()
        }
        analysis_results['geographic_distribution'] = geo_dist
        
        # Occupation analysis
        occupation_dist = {
            'status': df['Actualmente te encuentras:'].value_counts(),
            'work_area': df['Si actualmente trabajas. ¿En qué área trabajas?'].value_counts()
        }
        analysis_results['occupation_analysis'] = occupation_dist
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Age distribution
        plt.subplot(2, 2, 1)
        sns.histplot(df['Edad:'])
        plt.title('Distribución de Edad')
        
        # Gender distribution
        plt.subplot(2, 2, 2)
        df['Género:'].value_counts().plot(kind='pie')
        plt.title('Distribución de Género')
        
        # Education level
        plt.subplot(2, 2, 3)
        sns.countplot(data=df, y='Grado de estudios:')
        plt.title('Nivel de Educación')
        
        # Socioeconomic level
        plt.subplot(2, 2, 4)
        sns.countplot(data=df, y='Nivel socioeconómico:')
        plt.title('Nivel Socioeconómico')
        
        plt.tight_layout()
        analysis_results['visualizations'] = plt.gcf()
        
        return analysis_results
    def save_eda(self, results):
        # Convert numpy/pandas objects to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.Index):
                return obj.tolist()
            return str(obj)

        # Create copy of results without matplotlib figure
        results_copy = results.copy()
        results_copy.pop('visualizations', None)
        
        # Save visualizations
        plt.savefig(os.path.join(self.analyzer.experiment_dir, 'eda_visualizations.png'))
        
        # Save results to JSON
        json_path = os.path.join(self.analyzer.experiment_dir, 'eda_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, default=convert_to_serializable, 
                    ensure_ascii=False, indent=4)
    def plot_results(self, results, algoritm):
        print("Visualizando resultados...")
        """Visualiza los resultados del modelo."""
        # Matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                annot=True, 
                fmt='d',
                xticklabels=self.analyzer.label_encoder.classes_,
                yticklabels=self.analyzer.label_encoder.classes_)
        plt.title(f'Matriz de Confusión - {algoritm} - {self.analyzer.pretrained_model_name}')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.savefig(os.path.join(self.analyzer.experiment_dir,f'matriz_confusion_{algoritm}.png'))
        # plt.show()

        # Resultados de validación cruzada
        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_scores'])
        plt.title(f'Validación Cruzada - {algoritm} - {self.analyzer.pretrained_model_name}')
        plt.ylabel('Puntuación')
        plt.savefig(os.path.join(self.analyzer.experiment_dir,f'validacion_cruzada_{algoritm}.png'))
        # plt.show()
    def save_train_test_data(self, X, y, file_prefix="dataset"):
        """
        Guarda las matrices X e y en archivos separados con un prefijo común.
        
        Args:
        - X (np.ndarray): Características.
        - y (np.ndarray): Etiquetas.
        - file_prefix (str): Prefijo para los archivos (sin extensión).
        """
        np.save(f"{file_prefix}_X.npy", X)
        np.save(f"{file_prefix}_y.npy", y)
        joblib.dump(self.analyzer.label_encoder, 'label_encoder.pkl')
        print(f"Datos guardados como {file_prefix}_X.npy y {file_prefix}_y.npy")
    def load_train_test_data(self, file_prefix="dataset"):
        """
        Carga las matrices X e y desde archivos guardados previamente.
        
        Args:
        - file_prefix (str): Prefijo de los archivos (sin extensión).
        
        Returns:
        - X (np.ndarray): Características cargadas.
        - y (np.ndarray): Etiquetas cargadas.
        """
        X = np.load(f"{file_prefix}_X.npy")
        y = np.load(f"{file_prefix}_y.npy")
        print(f"Datos cargados desde {file_prefix}_X.npy y {file_prefix}_y.npy")
        label_encoder = joblib.load('label_encoder.pkl')
        return X, y, label_encoder
