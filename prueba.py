import pandas as pd
import numpy as np

def check_csv(path, nombre):
    """Imprime estadísticas básicas del CSV."""
    df = pd.read_csv(path, header=0)
    arr = df.values  # array 2D
    print(f"\n=== Estadísticas para {nombre} ===")
    print("Shape:", arr.shape)
    print("Min:", np.min(arr))
    print("Max:", np.max(arr))
    print("Mean:", np.mean(arr))
    print("Std:", np.std(arr))

# Uso:
check_csv("resultados/align parallel 1/preprocesada/matriz_3D_aplanada.csv", "CSV1 (Severidad 0)")
check_csv("resultados/align parallel 2/preprocesada/matriz_3D_aplanada.csv", "CSV2 (Severidad 1)")
check_csv("resultados/align parallel 3/preprocesada/matriz_3D_aplanada.csv", "CSV3 (Severidad 2)")
check_csv("resultados/align parallel 4/preprocesada/matriz_3D_aplanada.csv", "CSV4 (Severidad 3)")
