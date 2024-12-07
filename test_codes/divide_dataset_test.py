import pandas as pd 


data = pd.read_csv("Textos_Dataset_Completo_utf8.csv")


# print(data.dtypes)

print(data["Género:"].unique())
print(data["Nivel socioeconómico:"].unique())
print(data["Edad:"].unique())
print(data["Grado de estudios:"].unique())


# Renombrar columnas para facilitar el manejo
data = data.rename(columns=lambda x: x.strip())

# Dividir por rangos de edad
age_groups = {
    "age_<=20": data[data["Edad:"] <= 20],
    "age_21_to_30": data[(data["Edad:"] > 20) & (data["Edad:"] <= 30)],
    "age_>30": data[data["Edad:"] > 30],
}

# Dividir por nivel de educación
education_levels = {
    "bachillerato": data[data["Grado de estudios:"].str.contains("Bachillerato", case=False, na=False)],
    "licenciatura": data[data["Grado de estudios:"].str.contains("Licenciatura", case=False, na=False)],
    "maestria": data[data["Grado de estudios:"].str.contains("Maestría", case=False, na=False)],
}

# Dividir por nivel socioeconómico
socioeconomic_levels = {
    "socio_bajo": data[data["Nivel socioeconómico:"].str.contains("Bajo", case=False, na=False)],
    "socio_medio": data[data["Nivel socioeconómico:"].str.contains("Medio", case=False, na=False)],
    "socio_alto": data[data["Nivel socioeconómico:"].str.contains("Alto", case=False, na=False)],
}


for group, df in {**age_groups, **education_levels, **socioeconomic_levels}.items():
    output_path = f"{group}.csv"
    df.to_csv(output_path, index=False)

# Listar los archivos generados
[f"{group}.csv" for group in {**age_groups, **education_levels, **socioeconomic_levels}]
