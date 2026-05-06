import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generar población (1,000 estudiantes)
np.random.seed(42)
poblacion = np.random.uniform(0, 5, 1000)
df = pd.DataFrame({'Promedio': poblacion})

# 2. Aplicar Muestreo por Juicio (Criterio: Promedio > 4.0)
# Este es nuestro "Juicio Experto"
muestra_juicio = df[df['Promedio'] > 4.0]

# 3. Seleccionar muestra final para análisis detallado
muestra_final = muestra_juicio.sample(n=50, random_state=42)

# 4. Cálculos para la validación
prom_pob = df['Promedio'].mean()
prom_muest = muestra_final['Promedio'].mean()

print(f"--- VALIDACIÓN DE RESULTADOS ---")
print(f"Promedio General (Población): {prom_pob:.2f}")
print(f"Promedio de Excelencia (Muestra): {prom_muest:.2f}")
print(f"Estudiantes que cumplen el criterio (>4.0): {len(muestra_juicio)}")

# 5. GENERACIÓN DE EVIDENCIA VISUAL (Pantallazo para Diapositiva 6)
plt.figure(figsize=(12, 5))

# Subplot 1: Distribución Global (Población)
plt.subplot(1, 2, 1)
sns.histplot(df['Promedio'], bins=20, color='skyblue', kde=True)
plt.axvline(prom_pob, color='red', linestyle='--', label=f'Media: {prom_pob:.2f}')
plt.title('Población Total (Sin Filtro)')
plt.xlabel('Promedio Académico')
plt.ylabel('Frecuencia de Estudiantes')
plt.legend()

# Subplot 2: Resultado del Muestreo por Juicio (Muestra)
plt.subplot(1, 2, 2)
sns.histplot(muestra_final['Promedio'], bins=10, color='gold', kde=True)
plt.axvline(prom_muest, color='red', linestyle='--', label=f'Media: {prom_muest:.2f}')
plt.title('Muestra por Juicio (Solo Excelencia > 4.0)')
plt.xlabel('Promedio Académico')
plt.ylabel('Frecuencia de Estudiantes')
plt.legend()

plt.tight_layout()
plt.show()