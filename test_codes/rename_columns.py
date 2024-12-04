import pandas as pd


nombre_columnas = {
    'Marca temporal': 'fecha',
    'ID Único': 'id',
    'Edad:': 'edad',
    'Género:': 'genero',
    'Nivel socioeconómico:': 'nivel_socioeconomico',
    'Grado de estudios:': 'grado_estudios',
    'Actualmente te encuentras:': 'situacion_actual',
    'Si actualmente trabajas. ¿En qué área trabajas?': 'area_trabajo',
    'Estado de origen:': 'estado_origen',
    'Municipio de origen:': 'municipio_origen',
    '1. Describa, ¿en qué situaciones últimamente ha sentido alegría?': 'pregunta_1',
    '2. Especifique, ¿en qué situaciones últimamente ha sentido ganas de llorar?': 'pregunta_2',
    '3. En las últimas dos semanas, ¿en qué momentos se ha sentido cansado?': 'pregunta_3',
    '4. ¿En qué situaciones de su día a día, puede identificar que se ha sentido preocupado?': 'pregunta_4',
    '5. Cuando la preocupación se hace presente en su vida, ¿cuáles son las sensaciones corporales que experimenta?': 'pregunta_5',
    '6. Si su escritor favorito le pidiera que le ayude a buscar el significado de la felicidad, ¿cuál sería la idea que usted escribiría?': 'pregunta_6',
    '7. Cuando tiene una elección importante que hacer, mencione ¿cuáles serían los pasos que seguiría para llegar a tomar una decisión?': 'pregunta_7',
    '8. Cuando sucede algo inesperado en su vida, por ejemplo: llegar tarde al trabajo, el automóvil se descompuso, se quedó sin batería el celular, olvidar las llaves, etc., describa ¿cómo reacciona ante estas situaciones?': 'pregunta_8',
    '9. Describa ¿En qué situaciones de su vida considera que se siente estresado o molesto?': 'pregunta_9',
    '10. Por favor, describa brevemente ¿qué hace cuando tiene que enfrentar una situación difícil?': 'pregunta_10'
}

# Lee el archivo CSV en un DataFrame
df = pd.read_csv('Textos_Dataset_Completo_utf8.csv')

# Renombra las columnas usando el diccionario
df = df.rename(columns=nombre_columnas)

# Muestra el DataFrame resultante
# print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Muestra información de columnas y tipos de datos
print(df.info())

df.to_csv('dataset_normalizado_utf8.csv', index=False)