import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene
from IPython.display import display

# Cargar los archivos CSV
df_fermento = pd.read_csv("datos_fermento.csv")  # Contiene pH, Brix y Glucosa final
df_glucosa = pd.read_csv("datos_glucosa.csv")    # Contiene mediciones de glucosa en días 0, 3, 6, 9, 12


# Definir función de estadística descriptiva
def estadistica_descriptiva(serie):
    return pd.DataFrame({
        'Media': [serie.mean()],
        'Mediana': [serie.median()],
        'Desviación estándar': [serie.std()],
        'Mínimo': [serie.min()],
        'Máximo': [serie.max()],
        'N': [serie.count()]
    })

# Reemplazar NaN en 'Repeticion' con 1
df_fermento['Repeticion'] = df_fermento['Repeticion'].fillna(1)

# Imputar NaN en 'pH' del tratamiento "Caña" con la media del grupo
caña_ph_mean = df_fermento[df_fermento['Tratamiento'] == 'Caña']['pH'].mean()
df_fermento.loc[(df_fermento['Tratamiento'] == 'Caña') & (df_fermento['pH'].isna()), 'pH'] = caña_ph_mean

# Imputar NaN en 'Glucosa' de "Cacao" y "Mango" con la media del grupo
cacao_gluc_mean = df_fermento[df_fermento['Tratamiento'] == 'Cacao']['Glucosa'].mean()
mango_gluc_mean = df_fermento[df_fermento['Tratamiento'] == 'Mango']['Glucosa'].mean()
df_fermento.loc[(df_fermento['Tratamiento'] == 'Cacao') & (df_fermento['Glucosa'].isna()), 'Glucosa'] = cacao_gluc_mean
df_fermento.loc[(df_fermento['Tratamiento'] == 'Mango') & (df_fermento['Glucosa'].isna()), 'Glucosa'] = mango_gluc_mean

# Reemplazar NaN en 'Glucosa_0' con 15 (valor inicial estándar)
df_glucosa['Glucosa_0'] = df_glucosa['Glucosa_0'].fillna(15)

# Imputar NaN en 'Glucosa_9' del tratamiento "Caña" con su media
caña_gluc9_mean = df_glucosa[df_glucosa['Tratamiento'] == 'Caña']['Glucosa_9'].mean()
df_glucosa.loc[(df_glucosa['Tratamiento'] == 'Caña') & (df_glucosa['Glucosa_9'].isna()), 'Glucosa_9'] = caña_gluc9_mean

# Imputar NaN en 'Glucosa_12' del tratamiento "Cacao" con su media
cacao_gluc12_mean = df_glucosa[df_glucosa['Tratamiento'] == 'Cacao']['Glucosa_12'].mean()
df_glucosa.loc[(df_glucosa['Tratamiento'] == 'Cacao') & (df_glucosa['Glucosa_12'].isna()), 'Glucosa_12'] = cacao_gluc12_mean


print("\nDatos limpios en df_fermento (NaN restantes):")
print(df_fermento.isna().sum())

print("\nDatos limpios en df_glucosa (NaN restantes):")
print(df_glucosa.isna().sum())
# ACTIVIDAD 2
variables = ['pH', 'Brix', 'Glucosa']

for var in variables:
    print(f"\n=== Estadísticas descriptivas para {var} ===")
    for tratamiento in df_fermento['Tratamiento'].unique():
        datos = df_fermento[df_fermento['Tratamiento'] == tratamiento][var]
        print(f"\nTratamiento: {tratamiento}")
        display(estadistica_descriptiva(datos))

plt.figure(figsize=(15, 5))

# ACTIVIDAD 3
for i, var in enumerate(variables, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x='Tratamiento', y=var, data=df_fermento)
    plt.title(f'Distribución de {var} por tratamiento')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ACTIVIDAD 4
# Prueba de normalidad (Shapiro-Wilk)
print("Prueba de normalidad (Shapiro-Wilk):")
for var in variables:
    print(f"\nVariable: {var}")
    for tratamiento in df_fermento['Tratamiento'].unique():
        datos = df_fermento[df_fermento['Tratamiento'] == tratamiento][var].dropna()
        stat, p = shapiro(datos)
        print(f"{tratamiento}: p-valor = {p:.4f} {'(Normal)' if p > 0.05 else '(No normal)'}")

# Prueba de homogeneidad de varianzas (Levene)
print("\nPrueba de homogeneidad de varianzas (Levene):")
for var in variables:
    grupos = [df_fermento[df_fermento['Tratamiento'] == t][var].dropna() for t in df_fermento['Tratamiento'].unique()]
    stat, p = levene(*grupos)
    print(f"{var}: p-valor = {p:.4f} {'(Varianzas homogéneas)' if p > 0.05 else '(Varianzas no homogéneas)'}")

# ACTIVIDAD 6
# a) Tiempo para G(t) < 0.1 g/L
G0 = 15
k = 0.155
target = 0.1

def tiempo_agotamiento(k, G0, target):
    return np.log(G0/target)/k

t_agotamiento = tiempo_agotamiento(k, G0, target)
print(f"La glucosa se agota (G < {target} g/L) a los {t_agotamiento:.2f} días")

# b) Gráfico de evolución de glucosa
t = np.linspace(0, t_agotamiento, 100)
G = G0 * np.exp(-k * t)

plt.figure(figsize=(10, 6))
plt.plot(t, G, label=f'G(t) = {G0} * exp(-{k} * t)')
plt.axhline(y=target, color='r', linestyle='--', label=f'Umbral ({target} g/L)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Concentración de glucosa (g/L)')
plt.title('Evolución de la glucosa durante la fermentación')
plt.legend()
plt.grid(True)
plt.show()