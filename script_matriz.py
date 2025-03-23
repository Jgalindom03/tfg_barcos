import os
import pandas as pd
import numpy as np
import h5py

def guardar_para_eda(matriz_3D, ruta_guardado):
    """
    Guarda la matriz 3D en formatos adecuados para análisis exploratorio:
    - HDF5 (almacenamiento eficiente y recarga rápida).
    - CSV (matriz aplanada).
    - CSV a partir de DataFrame.
    """
    # Crear la carpeta de guardado si no existe
    os.makedirs(ruta_guardado, exist_ok=True)

    # Guardar en HDF5
    hdf5_path = os.path.join(ruta_guardado, "matriz_3D.h5")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("matriz_3D", data=matriz_3D)
    print(f"Matriz guardada en formato HDF5: {hdf5_path}")

    # Aplanar la matriz 3D a 2D
    matriz_aplanada = matriz_3D.reshape(-1, matriz_3D.shape[2])

    # Guardar en CSV (matriz aplanada)
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
    Lee todos los archivos CSV de la carpeta indicada y conforma una matriz 3D,
    sin asumir un número fijo de archivos ni de filas/columnas.

    Pasos:
    ------
    1. Lista y ordena todos los archivos CSV de la carpeta.
    2. Carga cada CSV como un DataFrame y lo convierte a un array NumPy.
    3. Verifica que todos los arrays tengan la misma forma.
    4. Apila (stack) todos los arrays en un único array 3D con forma:
         (n_archivos, n_filas, n_columnas).

    Retorna
    -------
    matriz_final : np.ndarray
        Matriz 3D con forma (n_archivos, n_filas, n_columnas).
    """
    # Listar y ordenar los archivos CSV de la carpeta
    archivos_csv = sorted([f for f in os.listdir(carpeta) if f.lower().endswith(".csv")])

    if not archivos_csv:
        raise ValueError(f"No se encontraron archivos CSV en la carpeta '{carpeta}'.")

    lista_matrices = []
    for archivo in archivos_csv:
        ruta_archivo = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta_archivo)

        # Convertir a numpy sin recortar ni exigir un tamaño fijo
        arr = df.to_numpy()
        lista_matrices.append(arr)

    # Verificar que todas las matrices tengan la misma forma
    shapes = [mat.shape for mat in lista_matrices]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(
            f"Los archivos CSV en la carpeta '{carpeta}' no tienen la misma forma. "
            f"Formas encontradas: {shapes}"
        )

    # Apilar para formar la matriz 3D: (n_archivos, n_filas, n_columnas)
    matriz_final = np.stack(lista_matrices, axis=0)
    print(f"Matriz final con forma {matriz_final.shape} generada correctamente para la carpeta '{carpeta}'.")
    return matriz_final


def procesar_carpetas(base_input=".", base_output="resultados"):
    """
    Busca y procesa todas las carpetas en 'base_input' y genera
    la matriz 3D para cada carpeta que contenga archivos CSV. Luego guarda la matriz
    en 'base_output'.

    Parámetros
    ----------
    base_input : str
        Directorio base donde se encuentran las carpetas a procesar.
    base_output : str
        Directorio donde se guardarán los resultados.
    """
    # Listar subcarpetas en base_input
    subcarpetas = [d for d in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, d))]

    if not subcarpetas:
        print(f"No se encontraron carpetas dentro de '{base_input}'.")
        return

    # Procesar cada carpeta encontrada
    for carpeta in subcarpetas:
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
    """
    Aquí hacemos la llamada para que:
      - base_input="severidad_alta":  directorio donde están los subdirectorios a procesar
      - base_output="severidad_alta/resultados": carpeta donde guardaremos la salida
    """
    procesar_carpetas(base_input="severidad_alta", base_output="severidad_alta/resultados")
