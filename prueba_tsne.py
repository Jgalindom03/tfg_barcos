import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuración global de estilos y tamaño de figura
sns.set(style="whitegrid", palette="muted", font_scale=0.9)
plt.rcParams["figure.figsize"] = (10, 6)


# =============================================================================
# Funciones para cálculo de hash y guardado de datos para EDA
# =============================================================================
def compute_md5(file_path):
    """
    Calcula el hash MD5 de un archivo para identificar cambios en su contenido.
    """
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


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
    try:
        import h5py  # Importar aquí si se usa este formato
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("matriz_3D", data=matriz_3D)
        print(f"Matriz guardada en formato HDF5: {hdf5_path}")
    except ImportError:
        print("h5py no está instalado. Se omite el guardado en HDF5.")

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

    La matriz intermedia se forma con forma (6, 30000, 7), donde cada elemento
    corresponde a un archivo. Luego se reorganiza para obtener la forma final:
        (n_experimentos, n_instancias, n_canales) = (7, 30000, 6)
    """
    archivos = sorted([f for f in os.listdir(carpeta) if f.lower().endswith(".csv")])
    
    if len(archivos) != 6:
        raise ValueError(f"Se esperaban 6 archivos CSV en la carpeta '{carpeta}', pero se encontraron {len(archivos)}.")
    
    lista_matrices = []
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta_archivo)
        df = df.head(30000)  # Tomar solo las primeras 30000 filas
        
        if df.shape[1] != 7:
            raise ValueError(f"El archivo {archivo} tiene {df.shape[1]} columnas; se esperaban 7.")
        
        lista_matrices.append(df.to_numpy())
    
    # Forma intermedia: (6, 30000, 7)
    matriz_stack = np.stack(lista_matrices, axis=0)
    # Reorganizar a la forma final: (7, 30000, 6)
    matriz_final = matriz_stack.transpose(2, 1, 0)
    
    print(f"Matriz final con forma {matriz_final.shape} generada correctamente para la carpeta '{carpeta}'.")
    return matriz_final


def procesar_align_parallel(base_input=".", base_output="resultados"):
    """
    Busca y procesa todas las carpetas en 'base_input' cuyo nombre comience con "align parallel".
    Para cada carpeta encontrada:
      1. Se carga y se forma la matriz 3D a partir de los archivos CSV.
      2. Se guardan los datos en formatos HDF5 y CSV dentro de 'base_output'.
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
            matriz_3D = cargar_y_formar_matriz(carpeta_input)
            carpeta_resultados = os.path.join(base_output, carpeta)
            guardar_para_eda(matriz_3D, carpeta_resultados)
        except Exception as e:
            print(f"Error al procesar la carpeta '{carpeta_input}': {e}")


# =============================================================================
# Funciones para TSNE
# =============================================================================
def aplicar_tsne(datos, n_components=2, perplexity=30, random_state=42):
    """
    Aplica TSNE a la matriz de datos y retorna la transformación.

    Parámetros:
    -----------
    datos : np.ndarray
        Matriz de datos de forma (n_muestras, n_features).
    n_components : int, opcional
        Número de dimensiones en la representación final (por defecto 2).
    perplexity : float, opcional
        Parámetro de TSNE (por defecto 30).
    random_state : int, opcional
        Semilla para reproducibilidad (por defecto 42).

    Retorna:
    --------
    datos_tsne : np.ndarray
        Datos transformados de forma (n_muestras, n_components).
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    datos_tsne = tsne.fit_transform(datos)
    return datos_tsne


def procesar_tsne_align_parallel(base_dir="resultados", sample_size=1000):
    """
    Para cada carpeta en 'base_dir' cuyo nombre comience con "align parallel", carga el archivo
    'matriz_3D_aplanada.csv', aplica TSNE y guarda un gráfico del resultado.

    Parámetros:
    -----------
    base_dir : str
        Directorio que contiene las carpetas a procesar (por defecto "resultados").
    sample_size : int, opcional
        Número de muestras a utilizar para TSNE (se realiza muestreo si el total es mayor).
    """
    carpetas = [d for d in os.listdir(base_dir)
                if d.startswith("align parallel") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not carpetas:
        print(f"No se encontraron carpetas en '{base_dir}' que comiencen con 'align parallel'.")
        return

    for carpeta in carpetas:
        csv_path = os.path.join(base_dir, carpeta, "matriz_3D_aplanada.csv")
        if not os.path.exists(csv_path):
            print(f"No se encontró el archivo '{csv_path}'.")
            continue

        try:
            df = pd.read_csv(csv_path)
            datos = df.values

            # Si hay muchas muestras, se realiza un muestreo para acelerar TSNE
            if datos.shape[0] > sample_size:
                indices = np.random.choice(datos.shape[0], sample_size, replace=False)
                datos_sample = datos[indices]
            else:
                datos_sample = datos

            datos_tsne = aplicar_tsne(datos_sample, n_components=2, perplexity=30, random_state=42)

            # Crear gráfico de dispersión
            plt.figure(figsize=(8, 6))
            plt.scatter(datos_tsne[:, 0], datos_tsne[:, 1], s=5, alpha=0.7)
            plt.title(f"TSNE - {carpeta}")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            plt.grid(True)

            plot_path = os.path.join(base_dir, carpeta, "tsne_plot.png")
            plt.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"TSNE plot guardado en: {plot_path}")
        except Exception as e:
            print(f"Error al procesar TSNE en la carpeta '{carpeta}': {e}")


# =============================================================================
# Función para exportar imágenes de análisis a partir de los CSV generados
# =============================================================================
def export_analysis_images(results_dir, output_dir="resultados_imagenes"):
    """
    Recorre todas las subcarpetas de 'results_dir' que contengan el archivo
    'matriz_3D_aplanada_dataframe.csv' y guarda los gráficos de análisis como imágenes (PNG)
    en subcarpetas dentro de 'output_dir'. Cada subcarpeta tendrá el nombre de la carpeta de origen.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir)
               if os.path.isdir(os.path.join(results_dir, d))]
    print(f"Se encontraron {len(subdirs)} subcarpetas en '{results_dir}'.")
    
    for subdir in subdirs:
        csv_file = os.path.join(subdir, "matriz_3D_aplanada_dataframe.csv")
        if not os.path.exists(csv_file):
            print(f"No se encontró {csv_file} en {subdir}. Se omite esta carpeta.")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error al cargar {csv_file}: {e}")
            continue

        # Imprimir información para verificar los datos
        print(f"\nProcesando: {csv_file}")
        print("Primeras 5 filas:")
        print(df.head())
        print("Dimensiones:", df.shape)
        file_hash = compute_md5(csv_file)
        print("Hash MD5 del CSV:", file_hash)
        
        folder_name = os.path.basename(subdir)
        folder_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)

        # ----- Imagen 1: Resumen, estadísticas y valores perdidos -----
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        resumen = (
            f"Carpeta: {folder_name}\n"
            f"CSV: {csv_file}\n"
            f"Dimensiones: {df.shape}\n\n"
            f"Primeras 5 filas:\n{df.head().to_string()}\n\n"
            f"Estadísticas Descriptivas:\n{df.describe().to_string()}\n\n"
            f"Valores perdidos:\n{df.isnull().sum().to_string()}"
        )
        ax.text(0.01, 0.99, resumen, verticalalignment="top", fontsize=7, family="monospace")
        fig_path = os.path.join(folder_output_dir, f"{folder_name}_summary.png")
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Seleccionar columnas para graficar (omitiendo 'archivo' si existe)
        plot_cols = [col for col in df.columns if col.lower() != "archivo"]

        # ----- Imagen 2: Histogramas combinados -----
        if plot_cols:
            n_hist = len(plot_cols)
            nrows = int(np.ceil(n_hist / 2))
            fig, axs = plt.subplots(nrows, 2, figsize=(10, 4*nrows))
            if nrows == 1:
                axs = np.array(axs).reshape(1, -1)
            axs = axs.flatten()
            for i, col in enumerate(plot_cols):
                sns.histplot(df[col], kde=True, bins=30, color="steelblue", ax=axs[i])
                axs[i].set_title(f"Histograma de {col}")
            for j in range(i+1, len(axs)):
                fig.delaxes(axs[j])
            plt.tight_layout()
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_histograms.png")
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        # ----- Imagen 3: Boxplots combinados -----
        if plot_cols:
            n_box = len(plot_cols)
            nrows = int(np.ceil(n_box / 2))
            fig, axs = plt.subplots(nrows, 2, figsize=(10, 4*nrows))
            if nrows == 1:
                axs = np.array(axs).reshape(1, -1)
            axs = axs.flatten()
            for i, col in enumerate(plot_cols):
                sns.boxplot(x=df[col], color="lightgreen", ax=axs[i])
                axs[i].set_title(f"Boxplot de {col}")
            for j in range(i+1, len(axs)):
                fig.delaxes(axs[j])
            plt.tight_layout()
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_boxplots.png")
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        # ----- Imagen 4: Heatmap de correlación -----
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Mapa de Calor de la Correlación")
        plt.tight_layout()
        fig_path = os.path.join(folder_output_dir, f"{folder_name}_heatmap.png")
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ----- Imagen 5: Pairplot -----
        try:
            pairgrid = sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.3})
            pairgrid.fig.suptitle(f"Pairplot ({folder_name})", y=1.02)
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_pairplot.png")
            pairgrid.fig.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(pairgrid.fig)
        except Exception as e:
            print(f"Error generando pairplot para {folder_name}: {e}")

        # ----- PCA: Preparar datos y asignar etiqueta 'archivo' -----
        if "archivo" not in df.columns:
            n_archivos = 6
            n_filas = df.shape[0]
            n_filas_archivo = n_filas // n_archivos
            lista_archivos = []
            for i in range(n_archivos):
                lista_archivos.extend([i+1] * n_filas_archivo)
            if len(lista_archivos) < n_filas:
                lista_archivos.extend([n_archivos] * (n_filas - len(lista_archivos)))
            df["archivo"] = lista_archivos

        pca_cols = [col for col in df.columns if col != "archivo"]
        X = df[pca_cols].values
        try:
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X)
        except Exception as e:
            print(f"Error aplicando PCA en {folder_name}: {e}")
            continue

        # ----- Imagen 6: PCA Scatter Plot (PC1 vs PC2) -----
        if X_pca.shape[1] >= 2:
            df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
            df_pca["archivo"] = df["archivo"]
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(x="PC1", y="PC2", hue="archivo", data=df_pca,
                            palette="Set1", alpha=0.5, ax=ax)
            ax.set_title("PCA: PC1 vs PC2")
            plt.tight_layout()
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_pca_scatter.png")
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        # ----- Imagen 7: PCA Varianza Acumulada -----
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance,
                marker="o", linestyle="--", color="b")
        ax.set_title("Varianza Acumulada Explicada por PCA")
        ax.set_xlabel("Número de Componentes")
        ax.set_ylabel("Varianza Acumulada")
        ax.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(folder_output_dir, f"{folder_name}_pca_variance.png")
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        print(f"Se han guardado las imágenes de análisis para {folder_name} en {folder_output_dir}")


# =============================================================================
# Bloque Principal
# =============================================================================
if __name__ == "__main__":
    # 1. Procesar carpetas "align parallel" para generar los CSV y otros formatos.
    procesar_align_parallel(base_input=".", base_output="resultados")
    
    # 2. Aplicar TSNE a los archivos CSV aplanados de cada carpeta "align parallel".
    procesar_tsne_align_parallel(base_dir="resultados", sample_size=1000)
    
    # 3. Exportar imágenes de análisis a partir de los CSV (método EDA completo).
    export_analysis_images(results_dir="resultados", output_dir="resultados_imagenes")
