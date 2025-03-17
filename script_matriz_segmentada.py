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
    archivos = sorted([f for f in os.listdir(carpeta) if f.lower().endswith(".csv")])
    
    if len(archivos) != 6:
        raise ValueError(f"Se esperaban 6 archivos CSV en la carpeta '{carpeta}', pero se encontraron {len(archivos)}.")
    
    lista_matrices = []
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta_archivo)
        df = df.head(30000)
        if df.shape[1] != 7:
            raise ValueError(f"El archivo {archivo} tiene {df.shape[1]} columnas; se esperaban 7.")
        lista_matrices.append(df.to_numpy())
    
    matriz_stack = np.stack(lista_matrices, axis=0)
    matriz_final = matriz_stack.transpose(2, 1, 0)
    
    print(f"Matriz final con forma {matriz_final.shape} generada correctamente para la carpeta '{carpeta}'.")
    return matriz_final

def segment_matrix_sliding_window(matrix, window_size, step):
    """
    Segmenta una matriz 3D usando un algoritmo de ventana deslizante.
    
    Parámetros:
        matrix (np.ndarray): Matriz de entrada de forma (n_experimentos, n_instancias, n_canales).
        window_size (int): Longitud de la ventana a lo largo del eje 'n_instancias'.
        step (int): Número de instancias para avanzar la ventana en cada paso.
    
    Retorna:
        np.ndarray: Matriz segmentada de forma (n_segments, n_experimentos, window_size, n_canales)
                   donde n_segments = (n_instancias - window_size) // step + 1.
    """
    n_experimentos, n_instancias, n_canales = matrix.shape
    segments = []
    
    for start in range(0, n_instancias - window_size + 1, step):
        segment = matrix[:, start:start + window_size, :]
        segments.append(segment)
    
    return np.stack(segments, axis=0)

def guardar_segmented_matrices(segmented_matrix, ruta_guardado):
    """
    Guarda la matriz segmentada en formatos HDF5 y CSV (aplanada) en la carpeta especificada.

    Parámetros:
        segmented_matrix (np.ndarray): Matriz segmentada de forma (n_segments, n_experimentos, window_size, n_canales).
        ruta_guardado (str): Carpeta donde se guardarán los archivos.
    """
    os.makedirs(ruta_guardado, exist_ok=True)
    
    # Guardar en HDF5
    hdf5_path = os.path.join(ruta_guardado, "matrices_segmentadas.h5")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("matrices_segmentadas", data=segmented_matrix)
    print(f"Matrices segmentadas guardadas en formato HDF5: {hdf5_path}")
    
    # Aplanar la matriz segmentada para CSV:
    # Combinamos las dimensiones n_segments y n_experimentos y aplanamos las dimensiones window_size y n_canales.
    n_segments, n_experimentos, window_size, n_canales = segmented_matrix.shape
    matriz_aplanada = segmented_matrix.reshape(n_segments * n_experimentos, window_size * n_canales)
    
    csv_path = os.path.join(ruta_guardado, "matrices_segmentadas_aplanadas.csv")
    np.savetxt(csv_path, matriz_aplanada, delimiter=",")
    print(f"Matrices segmentadas guardadas en formato CSV aplanado: {csv_path}")

def procesar_align_parallel(base_input=".", base_output="resultados", window_size=1000, step=500):
    """
    Busca y procesa todas las carpetas en 'base_input' cuyo nombre comience con "align parallel".
    Para cada carpeta:
      1. Se carga y forma la matriz 3D a partir de los archivos CSV.
      2. Se guardan los datos en formatos HDF5 y CSV en una carpeta dentro de 'base_output'.
      3. Se segmenta la matriz 3D usando una ventana deslizante.
      4. Se guarda la matriz segmentada en una subcarpeta llamada "matrices_segmentadas".
    
    Parámetros:
    -----------
    base_input : str, opcional
        Directorio base donde se encuentran las carpetas a procesar.
    base_output : str, opcional
        Directorio donde se guardarán los resultados.
    window_size : int, opcional
        Tamaño de la ventana para la segmentación (por defecto 1000).
    step : int, opcional
        Paso para la ventana deslizante (por defecto 500).
    """
    carpetas = [d for d in os.listdir(base_input)
                if d.startswith("align parallel") and os.path.isdir(os.path.join(base_input, d))]
    
    if not carpetas:
        print(f"No se encontraron carpetas que comiencen con 'align parallel' en {base_input}.")
        return
    
    for carpeta in carpetas:
        carpeta_input = os.path.join(base_input, carpeta)
        print(f"\nProcesando la carpeta: {carpeta_input}")
        try:
            # Cargar y formar la matriz 3D
            matriz_3D = cargar_y_formar_matriz(carpeta_input)
            
            # Crear carpeta para resultados de la matriz original
            carpeta_resultados = os.path.join(base_output, carpeta)
            guardar_para_eda(matriz_3D, carpeta_resultados)
            
            # Realizar la segmentación mediante ventana deslizante
            segmentos = segment_matrix_sliding_window(matriz_3D, window_size, step)
            print(f"Segmentos generados con forma: {segmentos.shape}")
            
            # Crear subcarpeta para guardar las matrices segmentadas
            carpeta_segmentadas = os.path.join(carpeta_resultados, "matrices_segmentadas")
            guardar_segmented_matrices(segmentos, carpeta_segmentadas)
            
        except Exception as e:
            print(f"Error al procesar la carpeta '{carpeta_input}': {e}")

if __name__ == "__main__":
    # Se pueden ajustar window_size y step según tus necesidades.
    procesar_align_parallel(base_input=".", base_output="resultados", window_size=1000, step=500)
