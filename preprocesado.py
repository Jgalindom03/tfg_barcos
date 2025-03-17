import os
import pandas as pd
import numpy as np
import h5py
import pywt
from scipy.signal import butter, filtfilt

#########################################
# 1. Funciones de Preprocesamiento
#########################################

# 1.1. Denoising

def wavelet_denoise(signal, wavelet='db4', level=1):
    """
    Realiza denoising de la señal usando transformada wavelet y thresholding suave.
    
    Parámetros:
        signal: np.ndarray
            Señal 1D a procesar.
        wavelet: str
            Tipo de wavelet a usar (por defecto 'db4').
        level: int
            Nivel de descomposición (por defecto 1).
            
    Retorna:
        Señal denoised reconstruida.
    """
    # Descomponer la señal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimar el ruido a partir del primer nivel de detalle
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    # Aplicar umbral a los coeficientes de detalle (no al de aproximación)
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    # Reconstruir la señal
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def butter_lowpass_filter(signal, cutoff, fs, order=5):
    """
    Aplica un filtro Butterworth pasa-bajo a la señal.
    
    Parámetros:
        signal: np.ndarray
            Señal 1D a filtrar.
        cutoff: float
            Frecuencia de corte (Hz).
        fs: float
            Frecuencia de muestreo (Hz).
        order: int
            Orden del filtro (por defecto 5).
    
    Retorna:
        Señal filtrada.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# 1.2. Detección de Outliers

def remove_outliers(signal, factor=1.5):
    """
    Detecta outliers en la señal usando el método del IQR y los reemplaza por NaN.
    
    Parámetros:
        signal: np.ndarray
            Señal 1D.
        factor: float
            Factor multiplicador del IQR (por defecto 1.5).
            
    Retorna:
        Señal con outliers reemplazados por NaN.
    """
    q1, q3 = np.percentile(signal, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    signal_no_outliers = np.where((signal < lower_bound) | (signal > upper_bound), np.nan, signal)
    return signal_no_outliers

# 1.3. Imputación de Datos Faltantes

def impute_missing(signal):
    """
    Imputa los valores faltantes (NaN) de la señal usando interpolación lineal.
    
    Parámetros:
        signal: np.ndarray
            Señal 1D con posibles NaN.
            
    Retorna:
        Señal con valores faltantes imputados.
    """
    s = pd.Series(signal)
    s_interpolated = s.interpolate(method='linear', limit_direction='both')
    return s_interpolated.values

# 1.4. Normalización

def normalize_signal(signal):
    """
    Normaliza la señal (standardization) restando la media y dividiendo por la desviación estándar.
    
    Parámetros:
        signal: np.ndarray
            Señal 1D.
            
    Retorna:
        Señal normalizada.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

# 1.5. Función que integra el preprocesamiento de la matriz 3D

def preprocess_matrix(matrix, fs=1000, cutoff=100, filter_method='wavelet'):
    """
    Preprocesa la matriz 3D (n_experimentos, n_instancias, n_canales) aplicando:
      1. Denoising (wavelet o Butterworth).
      2. Detección y eliminación de outliers.
      3. Imputación de datos faltantes.
      4. Normalización.
      
    Parámetros:
        matrix: np.ndarray
            Matriz de forma (n_experimentos, n_instancias, n_canales).
        fs: float
            Frecuencia de muestreo (para Butterworth).
        cutoff: float
            Frecuencia de corte (para Butterworth).
        filter_method: str
            Método de denoising a usar: 'wavelet' o 'butterworth'.
    
    Retorna:
        processed_matrix: np.ndarray
            Matriz preprocesada con la misma forma.
    """
    n_experimentos, n_instancias, n_canales = matrix.shape
    processed_matrix = np.empty_like(matrix)
    
    for exp in range(n_experimentos):
        for ch in range(n_canales):
            signal = matrix[exp, :, ch]
            
            # 1. Denoising
            if filter_method == 'wavelet':
                denoised_signal = wavelet_denoise(signal)
            elif filter_method == 'butterworth':
                denoised_signal = butter_lowpass_filter(signal, cutoff, fs)
            else:
                denoised_signal = signal  # Sin aplicar ningún filtro
                
            # 2. Detección de outliers
            signal_no_outliers = remove_outliers(denoised_signal)
            
            # 3. Imputación de datos faltantes
            imputed_signal = impute_missing(signal_no_outliers)
            
            # 4. Normalización
            normalized_signal = normalize_signal(imputed_signal)
            
            processed_matrix[exp, :, ch] = normalized_signal
    return processed_matrix

#########################################
# 2. Funciones de Manejo de la Matriz
#########################################

def guardar_para_eda(matriz_3D, ruta_guardado):
    """
    Guarda la matriz 3D en formatos adecuados para análisis exploratorio:
      - HDF5 (almacenamiento eficiente).
      - CSV (matriz aplanada).
      - CSV a partir de DataFrame.
    """
    os.makedirs(ruta_guardado, exist_ok=True)

    # Guardar en HDF5
    hdf5_path = os.path.join(ruta_guardado, "matriz_3D.h5")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("matriz_3D", data=matriz_3D)
    print(f"Matriz guardada en formato HDF5: {hdf5_path}")

    # Aplanar la matriz 3D a 2D: (n_experimentos * n_instancias, n_canales)
    matriz_aplanada = matriz_3D.reshape(-1, matriz_3D.shape[2])
    csv_path = os.path.join(ruta_guardado, "matriz_3D_aplanada.csv")
    header = ",".join([f"Ch{i+1}" for i in range(matriz_3D.shape[2])])
    np.savetxt(csv_path, matriz_aplanada, delimiter=",", header=header, comments="")
    print(f"Matriz guardada en formato CSV aplanado: {csv_path}")

    # Guardar también como DataFrame CSV
    df = pd.DataFrame(matriz_aplanada, columns=[f"Ch{i+1}" for i in range(matriz_3D.shape[2])])
    df_path = os.path.join(ruta_guardado, "matriz_3D_aplanada_dataframe.csv")
    df.to_csv(df_path, index=False)
    print(f"Matriz guardada como DataFrame CSV: {df_path}")

def cargar_y_formar_matriz(carpeta):
    """
    Lee los 6 archivos CSV de la carpeta indicada y conforma una matriz 3D.
    
    Se asume que cada CSV tiene al menos 30000 filas y 7 columnas.
    La forma final de la matriz es: (n_experimentos, n_instancias, n_canales) = (7, 30000, 6)
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
    
    # Forma intermedia: (6, 30000, 7)
    matriz_stack = np.stack(lista_matrices, axis=0)
    # Reorganizar a forma final: (7, 30000, 6)
    matriz_final = matriz_stack.transpose(2, 1, 0)
    print(f"Matriz final con forma {matriz_final.shape} generada correctamente para la carpeta '{carpeta}'.")
    return matriz_final

def procesar_align_parallel(base_input=".", base_output="resultados", fs=1000, cutoff=100, filter_method='wavelet'):
    """
    Busca y procesa todas las carpetas en 'base_input' cuyo nombre comience con "align parallel".
    Para cada carpeta se realizan los siguientes pasos:
      1. Carga y conformación de la matriz 3D.
      2. Guardado de la matriz original para EDA.
      3. Preprocesamiento de la matriz (denoising, outlier detection, imputación, normalización).
      4. Guardado de la matriz preprocesada.
    
    Parámetros:
        base_input: str
            Directorio base donde se encuentran las carpetas.
        base_output: str
            Directorio donde se guardarán los resultados.
        fs: float
            Frecuencia de muestreo para el filtro Butterworth.
        cutoff: float
            Frecuencia de corte para el filtro Butterworth.
        filter_method: str
            Método de denoising: 'wavelet' o 'butterworth'.
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
            # 1. Cargar y formar la matriz 3D
            matriz_3D = cargar_y_formar_matriz(carpeta_input)
            
            # 2. Guardar la matriz original para análisis exploratorio
            carpeta_resultados = os.path.join(base_output, carpeta)
            guardar_para_eda(matriz_3D, carpeta_resultados)
            
            # 3. Preprocesamiento de la matriz
            matriz_preprocesada = preprocess_matrix(matriz_3D, fs=fs, cutoff=cutoff, filter_method=filter_method)
            
            # 4. Guardar la matriz preprocesada en una subcarpeta
            carpeta_resultados_preproc = os.path.join(base_output, carpeta, "preprocesada")
            os.makedirs(carpeta_resultados_preproc, exist_ok=True)
            guardar_para_eda(matriz_preprocesada, carpeta_resultados_preproc)
            
        except Exception as e:
            print(f"Error al procesar la carpeta '{carpeta_input}': {e}")

#########################################
# 3. Ejecución Principal
#########################################

if __name__ == "__main__":
    # Parámetros de ejemplo:
    # - base_input: directorio donde se encuentran las carpetas "align parallel"
    # - base_output: directorio donde se guardarán los resultados
    # - fs: frecuencia de muestreo (Hz)
    # - cutoff: frecuencia de corte para el filtro Butterworth (si se usa)
    # - filter_method: elegir 'wavelet' o 'butterworth'
    
    procesar_align_parallel(base_input=".", 
                             base_output="resultados", 
                             fs=1000, 
                             cutoff=100, 
                             filter_method='wavelet')
