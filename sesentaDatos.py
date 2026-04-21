import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. CARGA Y ORGANIZACIÓN DE DATOS (Muestra n=60)
# ---------------------------------------------------------
datos_eficiencia = [
    70, 72, 75, 75, 78, 80, 80, 81, 82, 82, 
    83, 83, 84, 84, 85, 85, 85, 86, 86, 86, 
    87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 
    90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 
    94, 94, 94, 95, 95, 95, 95, 96, 96, 96, 
    97, 97, 97, 98, 98, 98, 99, 99, 99, 99
]

# Convertimos a un DataFrame de Pandas para análisis profesional
df = pd.DataFrame(datos_eficiencia, columns=['Eficiencia'])

# 2. CÁLCULO DE ESTADÍSTICOS DESCRIPTIVOS
# ---------------------------------------------------------
media = df['Eficiencia'].mean()
mediana = df['Eficiencia'].median()
moda = df['Eficiencia'].mode().values.tolist()
desviacion = df['Eficiencia'].std()
varianza = df['Eficiencia'].var()
cv = (desviacion / media) * 100
rango = df['Eficiencia'].max() - df['Eficiencia'].min()

# Cuartiles y RIC
q1 = df['Eficiencia'].quantile(0.25)
q3 = df['Eficiencia'].quantile(0.75)
ric = q3 - q1

# Medidas de Forma
asimetria = df['Eficiencia'].skew()
curtosis = df['Eficiencia'].kurtosis()

# 3. IMPRESIÓN DE RESULTADOS TÉCNICOS
# ---------------------------------------------------------
print("-" * 40)
print("   INFORME ESTADÍSTICO DE EFICIENCIA")
print("-" * 40)
print(f"Media Aritmética:   {media:.2f}")
print(f"Mediana:            {mediana:.2f}")
print(f"Moda:               {moda}")
print(f"Desviación Estándar:{desviacion:.2f}")
print(f"Varianza:           {varianza:.2f}")
print(f"Coef. de Variación: {cv:.2f}%")
print(f"Rango Total:        {rango}")
print(f"Rango Interc. (RIC): {ric}")
print(f"Asimetría (Sesgo):  {asimetria:.4f}")
print(f"Curtosis:           {curtosis:.4f}")
print("-" * 40)

# 4. VISUALIZACIÓN DE DATOS (BOXPLOT E HISTOGRAMA)
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpio para ingeniería
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histograma con curva KDE (Densidad)
sns.histplot(df['Eficiencia'], kde=True, ax=ax1, color='royalblue', bins=6)
ax1.set_title('Distribución de Frecuencias e Histograma', fontsize=14)
ax1.set_xlabel('Porcentaje de Eficiencia (%)')
ax1.set_ylabel('Frecuencia de Observaciones')
ax1.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
ax1.axvline(mediana, color='green', linestyle='-', label=f'Mediana: {mediana:.2f}')
ax1.legend()

# Diagrama de Caja y Bigotes (Boxplot) para análisis de Outliers
sns.boxplot(x=df['Eficiencia'], ax=ax2, color='lightcoral', width=0.5)
ax2.set_title('Diagrama de Caja y Bigotes (Boxplot)', fontsize=14)
ax2.set_xlabel('Porcentaje de Eficiencia (%)')

# Añadir etiquetas de los cuartiles en el gráfico
ax2.text(q1, 0.2, f'Q1: {q1}', horizontalalignment='center', fontweight='bold')
ax2.text(mediana, -0.3, f'Med: {mediana}', horizontalalignment='center', fontweight='bold')
ax2.text(q3, 0.2, f'Q3: {q3}', horizontalalignment='center', fontweight='bold')

plt.tight_layout()
plt.show()