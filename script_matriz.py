import os
import pandas as pd
import numpy as np
import h5py

def guardar_para_eda(matriz_3D, ruta_guardado):
    """
    Guarda la matriz 3D en formatos adecuados para análisis exploratorio:
    - HDF5 (almacenamiento eficiente y recarga rápida).
    - CSV (matriz aplanada).
    - CSV a partir de DataFrame (opcional).

    Parámetros:
    -----------
    matriz_3D : np.ndarray
        Matriz 3D con forma (n_experimentos, n_instancias, n_canales).
    ruta_guardado : str
        Carpeta donde se guardarán los archivos.
    """
    # Crear la carpeta de guardado si no existe
    os.makedirs(ruta_guardado, exist_ok=True)

    # Guardar en HDF5
    hdf5_path = os.path.join(ruta_guardado, "matriz_3D.h5")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("matriz_3D", data=matriz_3D)
    print(f"Matriz guardada en formato HDF5: {hdf5_path}")

    # Aplanar la matriz 3D a 2D:
    # La matriz aplanada tendrá forma: (n_experimentos * n_instancias, n_canales)
    matriz_aplanada = matriz_3D.reshape(-1, matriz_3D.shape[2])

    # Guardar en CSV
    csv_path = os.path.join(ruta_guardado, "matriz_3D_aplanada.csv")
    header = ",".join([f"Ch{i+1}" for i in range(matriz_3D.shape[2])])
    np.savetxt(csv_path, matriz_aplanada, delimiter=",", header=header, comments="")
    print(f"Matriz guardada en formato CSV aplanado: {csv_path}")

    # Guardar también como un DataFrame CSV
    df = pd.DataFrame(matriz_aplanada, columns=[f"Ch{i+1}" for i in range(matriz_3D.shape[2])])
    df_path = os.path.join(ruta_guardado, "matriz_3D_aplanada_dataframe.csv")
    df.to_csv(df_path, index=False)
    print(f"Matriz guardada como DataFrame CSV: {df_path}")

def cargar_y_formar_matriz(carpeta):
    """
    Lee los 6 archivos CSV de la carpeta indicada y conforma una matriz 3D.

    Se asume que cada CSV puede tener más de 30000 filas.
    Para cada archivo, se toma únicamente la porción de las primeras 30000 filas y
    se comprueba que tenga 7 columnas.

    La matriz intermedia se formará con forma (6, 30000, 7), donde cada elemento de la lista corresponde
    a un archivo. Luego se reorganiza para obtener la forma final:
        (n_experimentos, n_instancias, n_canales) = (7, 30000, 6)
    """
    # Listar y ordenar los archivos CSV de la carpeta
    archivos = sorted([f for f in os.listdir(carpeta) if f.lower().endswith(".csv")])
    
    if len(archivos) != 6:
        raise ValueError(f"Se esperaban 6 archivos CSV en la carpeta '{carpeta}', pero se encontraron {len(archivos)}.")
    
    lista_matrices = []
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta_archivo)
        
        # Tomar solo las primeras 30000 filas
        df = df.head(30000)
        
        if df.shape[1] != 7:
            raise ValueError(f"El archivo {archivo} tiene {df.shape[1]} columnas; se esperaban 7.")
        
        lista_matrices.append(df.to_numpy())
    
    # Forma intermedia: (6, 30000, 7)  --> 6 archivos (canales), cada uno con 30000 filas y 7 columnas
    matriz_stack = np.stack(lista_matrices, axis=0)
    
    # Reorganizar a la forma final: (n_experimentos, n_instancias, n_canales) = (7, 30000, 6)
    matriz_final = matriz_stack.transpose(2, 1, 0)
    
    print(f"Matriz final con forma {matriz_final.shape} generada correctamente para la carpeta '{carpeta}'.")
    return matriz_final

def procesar_align_parallel(base_input=".", base_output="resultados"):
    """
    Busca y procesa todas las carpetas en 'base_input' cuyo nombre comience con "align parallel".
    Para cada carpeta encontrada, se realiza lo siguiente:
      1. Se carga y se forma la matriz 3D a partir de los archivos CSV contenidos en la carpeta.
      2. Se guardan los datos en formatos HDF5 y CSV dentro de una carpeta en 'base_output'
         que conserva el nombre de la carpeta de origen.
    
    Parámetros:
    -----------
    base_input : str, opcional
        Directorio base donde se encuentran las carpetas a procesar (por defecto el directorio actual ".").
    base_output : str, opcional
        Directorio donde se guardarán los resultados (por defecto "resultados").
    
    Ejemplo de uso:
    ---------------
        procesar_align_parallel(base_input="datos", base_output="resultados")
    """
    # Listar todas las carpetas que comiencen con "align parallel" en el directorio base_input
    carpetas = [d for d in os.listdir(base_input)
                if d.startswith("align parallel") and os.path.isdir(os.path.join(base_input, d))]
    
    if not carpetas:
        print(f"No se encontraron carpetas que comiencen con 'align parallel' en {base_input}.")
        return
    
    # Procesar cada carpeta encontrada
    for carpeta in carpetas:
        carpeta_input = os.path.join(base_input, carpeta)
        print(f"\nProcesando la carpeta: {carpeta_input}")
        try:
            # Cargar y formar la matriz 3D
            matriz_3D = cargar_y_formar_matriz(carpeta_input)
            # Crear carpeta de resultados para la carpeta actual
            carpeta_resultados = os.path.join(base_output, carpeta)
            guardar_para_eda(matriz_3D, carpeta_resultados)
        except Exception as e:
            print(f"Error al procesar la carpeta '{carpeta_input}': {e}")

if __name__ == "__main__":
    # Llama a la función para procesar todas las carpetas "align parallel"
    procesar_align_parallel(base_input=".", base_output="resultados")
