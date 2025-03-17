import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import hashlib

# Configuración global de estilos y tamaño de figura
sns.set(style="whitegrid", palette="muted", font_scale=0.9)
plt.rcParams["figure.figsize"] = (10, 6)

def compute_md5(file_path):
    """
    Calcula el hash MD5 de un archivo para identificar cambios en su contenido.
    """
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def export_analysis_images(results_dir, output_dir="resultados_imagenes"):
    """
    Recorre todas las subcarpetas de 'results_dir' que contengan el archivo
    'matriz_3D_aplanada_dataframe.csv' y guarda cada uno de los gráficos del análisis
    como imágenes (PNG) en subcarpetas dentro de 'output_dir'. Cada subcarpeta tendrá el nombre
    de la carpeta de origen.
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

        # Imprimir información para verificar que los datos sean distintos
        print(f"\nProcesando: {csv_file}")
        print("Primeras 5 filas:")
        print(df.head())
        print("Dimensiones:", df.shape)
        file_hash = compute_md5(csv_file)
        print("Hash MD5 del CSV:", file_hash)
        
        folder_name = os.path.basename(subdir)
        # Crear subcarpeta de salida para este conjunto de resultados
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
        if len(plot_cols) > 0:
            n_hist = len(plot_cols)
            nrows = int(np.ceil(n_hist / 2))
            fig, axs = plt.subplots(nrows, 2, figsize=(10, 4*nrows))
            if nrows == 1:
                axs = np.array(axs).reshape(1, -1)
            axs = axs.flatten()
            for i, col in enumerate(plot_cols):
                sns.histplot(df[col], kde=True, bins=30, color="steelblue", ax=axs[i])
                axs[i].set_title(f"Histograma de {col}")
            # Eliminar ejes vacíos
            for j in range(i+1, len(axs)):
                fig.delaxes(axs[j])
            plt.tight_layout()
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_histograms.png")
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        # ----- Imagen 3: Boxplots combinados -----
        if len(plot_cols) > 0:
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

        # ----- Imagen 5: Pairplot (si es posible) -----
        try:
            pairgrid = sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.3})
            pairgrid.fig.suptitle(f"Pairplot ({folder_name})", y=1.02)
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_pairplot.png")
            pairgrid.fig.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(pairgrid.fig)
        except Exception as e:
            print(f"Error generando pairplot para {folder_name}: {e}")

        # ----- PCA: Preparar datos -----
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
        '''
        Imagen 6: PCA Scatter Plot (PC1 vs PC2) con 6 líneas (una por cada entrada) 
        if X_pca.shape[1] >= 2:
            df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
            df_pca["archivo"] = df["archivo"]
            fig, ax = plt.subplots(figsize=(8,6))
            # Iteramos sobre cada valor único en la columna "archivo"
            for valor in sorted(df_pca["archivo"].unique()):
                subset = df_pca[df_pca["archivo"] == valor]
                # Se dibujan tanto los puntos como una línea que conecta dichos puntos
                ax.plot(subset["PC1"], subset["PC2"], marker='o', linestyle='-', label=f"Entrada {valor}", alpha=0.7)
            ax.set_title("PCA: PC1 vs PC2")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(title="Entrada")
            plt.tight_layout()
            fig_path = os.path.join(folder_output_dir, f"{folder_name}_pca_scatter.png")
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
        '''
    
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

        # ----- Imagen 7: PCA Varianza Acumulada explicada por grupo (cada "archivo") -----
        fig, ax = plt.subplots(figsize=(8,6))
        # Iteramos sobre cada grupo identificado en la columna "archivo"
        for valor in sorted(df["archivo"].unique()):
            subset = df[df["archivo"] == valor]
            X_sub = subset[pca_cols].values
            try:
                pca_sub = PCA(n_components=0.95)
                pca_sub.fit(X_sub)
            except Exception as e:
                print(f"Error aplicando PCA para archivo {valor} en {folder_name}: {e}")
                continue
            explained_variance = pca_sub.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            ax.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance,
                    marker="o", linestyle="--", label=f"Entrada {valor}")
            
        ax.set_title("Varianza Acumulada Explicada por PCA")
        ax.set_xlabel("Número de Componentes")
        ax.set_ylabel("Varianza Acumulada")
        ax.grid(True)
        ax.legend(title="Entrada")
        plt.tight_layout()
        fig_path = os.path.join(folder_output_dir, f"{folder_name}_pca_variance.png")
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        print(f"Se han guardado las imágenes de análisis para {folder_name} en {folder_output_dir}")


if __name__ == "__main__":
    # 'resultados' es la carpeta base donde se encuentran las subcarpetas con los CSV.
    base_folder = "resultados"
    export_analysis_images(base_folder, output_dir="resultados_imagenes")
