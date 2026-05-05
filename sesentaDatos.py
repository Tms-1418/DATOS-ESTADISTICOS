import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. DATOS
datos_eficiencia = [
    70, 72, 75, 75, 78, 80, 80, 81, 82, 82, 
    83, 83, 84, 84, 85, 85, 85, 86, 86, 86, 
    87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 
    90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 
    94, 94, 94, 95, 95, 95, 95, 96, 96, 96, 
    97, 97, 97, 98, 98, 98, 99, 99, 99, 99
]

df = pd.DataFrame(datos_eficiencia, columns=['Eficiencia'])
n = len(df)

# 2. ESTADÍSTICA DESCRIPTIVA (CONSISTENTE CON TU INFORME)
media = df['Eficiencia'].mean()
mediana = df['Eficiencia'].median()
moda = df['Eficiencia'].mode().values.tolist()

# Usamos ddof=0 para que coincida con tu informe (poblacional)
desviacion = df['Eficiencia'].std(ddof=0)
varianza = df['Eficiencia'].var(ddof=0)

cv = (desviacion / media) * 100
rango = df['Eficiencia'].max() - df['Eficiencia'].min()

q1 = df['Eficiencia'].quantile(0.25)
q3 = df['Eficiencia'].quantile(0.75)
ric = q3 - q1

asimetria = df['Eficiencia'].skew()
curtosis = df['Eficiencia'].kurtosis()

# 3. PRUEBA DE HIPÓTESIS (Z)
mu0 = 90
alpha = 0.05

error_std = desviacion / np.sqrt(n)
Z = (media - mu0) / error_std

# Valor-p (bilateral)
p_valor = 2 * (1 - stats.norm.cdf(abs(Z)))

# Decisión
rechazar = abs(Z) > 1.96

# 4. INTERVALO DE CONFIANZA (95%)
z_crit = 1.96
lim_inf = media - z_crit * error_std
lim_sup = media + z_crit * error_std

# 5. RESULTADOS
print("-" * 50)
print(" INFORME ESTADÍSTICO COMPLETO ")
print("-" * 50)

print(f"Media:                {media:.2f}")
print(f"Mediana:              {mediana:.2f}")
print(f"Moda:                 {moda}")
print(f"Desviación estándar:  {desviacion:.2f}")
print(f"Varianza:             {varianza:.2f}")
print(f"Coef. Variación:      {cv:.2f}%")
print(f"Rango:                {rango}")
print(f"RIC:                  {ric}")
print(f"Asimetría:            {asimetria:.4f}")
print(f"Curtosis:             {curtosis:.4f}")

print("\n--- PRUEBA DE HIPÓTESIS ---")
print(f"H0: μ = {mu0}")
print(f"Z calculado:          {Z:.4f}")
print(f"Valor-p:              {p_valor:.4f}")

if rechazar:
    print("Decisión: RECHAZAR H0")
else:
    print("Decisión: NO RECHAZAR H0")

print("\n--- INTERVALO DE CONFIANZA 95% ---")
print(f"({lim_inf:.2f}, {lim_sup:.2f})")

# 6. GRÁFICAS
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histograma
sns.histplot(df['Eficiencia'], kde=True, ax=ax1, bins=6)
ax1.set_title('Histograma de Eficiencia')
ax1.axvline(media, linestyle='--', label=f'Media: {media:.2f}')
ax1.axvline(mu0, linestyle=':', label='Valor teórico: 90')
ax1.legend()

# Boxplot
sns.boxplot(x=df['Eficiencia'], ax=ax2)
ax2.set_title('Diagrama de Caja')

plt.tight_layout()
plt.show()
