# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:17:51 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, shapiro

# Datos
data = [2.2, 7.6, 2.9, 4.6, 4.1, 3.9, 7.4, 3.2, 5.1, 5.3, 20.1, 2.3, 5.5, 32.7, 9.1, 1.7, 3.2, 5.8, 16.3, 15.9, 5.9, 6.7, 3.4, 40.5]

# Crear dataframe
df = pd.DataFrame(data, columns=['Tasa de Incremento'])

# Funciones de transformación
def transform_data(data, transformation):
    if transformation == 'y = x^2':
        return np.power(data, 2)
    elif transformation == 'y = sqrt(x)':
        return np.sqrt(data)
    elif transformation == 'y = ln(x)':
        return np.log(data)
    elif transformation == 'y = 1/x':
        return 1 / data
    else:
        return data

# Título y descripción de los datos
st.title('Transformación de Datos para Aumentar la Simetría')
st.write('Se consideran los siguientes datos, correspondientes a la tasa de incrementos de precios al consumo, en 2015, para 25 países de la OCDE:')
st.write('2.2, 7.6, 2.9, 4.6, 4.1, 3.9, 7.4, 3.2, 5.1, 5.3, 20.1, 2.3, 5.5, 32.7, 9.1, 1.7, 3.2, 5.8, 16.3, 15.9, 5.9, 6.7, 3.4, 40.5')
st.write('Transforma los datos para aumentar la simetría.')

# Sección de explicación en el sidebar
with st.sidebar:
    st.header("Explicación del Análisis")
    st.subheader("1. Enunciado de H0")
    st.write("H0: Los datos provienen de una distribución normal.")
    
    st.subheader("2. Cálculo del Estadístico de Shapiro-Wilk")
    st.write("""
    El estadístico de Shapiro-Wilk se calcula utilizando la fórmula:

    W = (sum(a_i * x_(i))^2) / sum((x_i - x̄)^2)
    
    donde x_(i) son los datos ordenados, x̄ es la media de la muestra, y a_i son coeficientes predefinidos.
    """)
    
    st.subheader("3. Función del Estadístico de Shapiro-Wilk")
    st.write("""
    El estadístico de Shapiro-Wilk mide la conformidad de una muestra con la distribución normal y se usa para:
    - Evaluar normalidad
    - Realizar pruebas de hipótesis
    """)
    
    st.subheader("4. Elección de Transformación")
    st.write("""
    La transformación que mejor ajusta los datos a una distribución normal se elige con base en el p-valor más alto obtenido de la prueba de Shapiro-Wilk.
    """)
    
    st.subheader("Resumen")
    st.write("""
    - **H0:** Los datos provienen de una distribución normal.
    - **Estadístico W:** Calcula la conformidad de los datos con la distribución normal.
    - **Función:** Evaluar normalidad y realizar pruebas de hipótesis.
    - **Elección de Transformación:** La transformación con el mayor p-valor en la prueba de Shapiro-Wilk es la que mejor ajusta los datos a una distribución normal.
    """)

# Visualizar datos originales
st.subheader('Datos Originales')
st.write(df)

# Graficar datos originales
st.subheader('Distribución de Datos Originales')
fig, ax = plt.subplots()
sns.histplot(df['Tasa de Incremento'], kde=True, ax=ax)
st.pyplot(fig)

# Calcular y mostrar sesgo original
original_skewness = skew(df['Tasa de Incremento'])
st.write(f'Sesgo de los datos originales: {original_skewness:.2f}')

# Combobox para elegir transformación
transformation = st.selectbox('Selecciona una transformación:', ['y = x^2', 'y = sqrt(x)', 'y = ln(x)', 'y = 1/x'])

# Transformar datos
transformed_data = transform_data(df['Tasa de Incremento'], transformation)
df['Datos Transformados'] = transformed_data

# Crear intervalos y calcular estadísticas para datos transformados
num_bins = 10  # Número de intervalos
freq, bins = np.histogram(transformed_data, bins=num_bins)
intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]

# Crear DataFrame para los datos transformados
df_transformed_intervals = pd.DataFrame({
    'Intervalo': [f'{interval[0]:.2f} - {interval[1]:.2f}' for interval in intervals],
    'Frecuencia': freq,
    'Tamaño del Intervalo': np.diff(bins)
})

# Mostrar los datos transformados
st.subheader('Datos Transformados')
st.write(df)

# Graficar datos transformados
st.subheader('Distribución de Datos Transformados')
fig, ax = plt.subplots()
sns.histplot(df['Datos Transformados'], kde=True, ax=ax)
st.pyplot(fig)

# Mostrar frecuencias de los datos transformados
st.subheader('Frecuencias de Datos Transformados')
st.write(df_transformed_intervals)

# Calcular y mostrar estadísticas de intervalos
medians = []
means = []
variances = []
std_devs = []
modes = []
ranges = []
shapiro_stats = []

for i in range(len(bins)-1):
    interval_data = transformed_data[(transformed_data >= bins[i]) & (transformed_data < bins[i+1])]
    medians.append(np.median(interval_data) if len(interval_data) > 0 else np.nan)
    means.append(np.mean(interval_data) if len(interval_data) > 0 else np.nan)
    variances.append(np.var(interval_data) if len(interval_data) > 0 else np.nan)
    std_devs.append(np.std(interval_data) if len(interval_data) > 0 else np.nan)
    modes.append(interval_data.mode().iloc[0] if len(interval_data) > 0 else np.nan)
    ranges.append(np.ptp(interval_data) if len(interval_data) > 0 else np.nan)
    
    # Solo calcular Shapiro-Wilk si hay al menos 3 datos en el intervalo
    if len(interval_data) >= 3:
        shapiro_stat, _ = shapiro(interval_data)
        shapiro_stats.append(shapiro_stat)
    else:
        shapiro_stats.append(np.nan)

df_intervals = pd.DataFrame({
    'Intervalo': intervals,
    'Frecuencia': freq,
    'Tamaño del Intervalo': np.diff(bins),
    'Mediana': medians,
    'Promedio': means,
    'Varianza': variances,
    'Desviación Estándar': std_devs,
    'Moda': modes,
    'Rango': ranges,
    'Estadístico de Shapiro-Wilk': shapiro_stats
})

# Reemplazar caracteres inválidos en el nombre de la hoja
def sanitize_sheet_name(name):
    invalid_chars = ['/', '\\', '?', '*', '[', ']']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

# Guardar los datos en un archivo Excel
filename = 'datos_transformados.xlsx'
with pd.ExcelWriter(filename) as writer:
    # Escribir datos originales
    df.to_excel(writer, sheet_name='Datos Originales', index=False)
    
    # Escribir datos transformados
    df_intervals.to_excel(writer, sheet_name=sanitize_sheet_name(transformation), index=False)

# Mostrar el enlace de descarga
with open(filename, 'rb') as f:
    st.download_button(
        label="Descargar Datos Transformados",
        data=f,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Determinar la mejor transformación
def find_best_transformation(data):
    transformations = ['y = x^2', 'y = sqrt(x)', 'y = ln(x)', 'y = 1/x']
    best_transformation = None
    best_p_value = 0
    
    for transformation in transformations:
        transformed_data = transform_data(data, transformation)
        _, p_value = shapiro(transformed_data)
        if p_value > best_p_value:
            best_p_value = p_value
            best_transformation = transformation
    
    return best_transformation, best_p_value

if st.button('Mostrar Mejor Transformación'):
    best_transformation, best_p_value = find_best_transformation(df['Tasa de Incremento'])
    st.write(f'La mejor transformación es: {best_transformation} con un p-valor de {best_p_value:.4f}')
