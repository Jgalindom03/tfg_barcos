
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

def cargar_y_formar_matriz(carpeta):
    """
    Lee todos los CSV de la carpeta indicada y devuelve una matriz 3D:
       (n_archivos, n_timesteps, n_canales)
    Asume que cada CSV tiene filas=timesteps, columnas=canales, 
    y descarta columna 'time' si existe.
    """
    archivos = sorted([
        f for f in os.listdir(carpeta) 
        if os.path.isfile(os.path.join(carpeta, f)) and f.lower().endswith(".csv")
    ])
    if not archivos:
        raise ValueError(f"No hay archivos CSV en '{carpeta}'")
    lista = []
    for fn in archivos:
        df = pd.read_csv(os.path.join(carpeta, fn))
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        lista.append(df.values)
    shapes = {m.shape for m in lista}
    if len(shapes) != 1:
        raise ValueError(f"Formas distintas en los CSV de '{carpeta}': {shapes}")
    matriz = np.stack(lista, axis=0)
    print(f"  • Matriz 3D cargada de '{carpeta}' → shape {matriz.shape}")
    return matriz

def eda_relations(matriz_3D, output_dir):
    """
    1) Estadísticos descriptivos por canal
    2) Histogramas y boxplots
    3) Matriz de correlación + heatmap
    4) Scatter (Ch1 vs Ch2)
    5) Cross-correlation entre Ch1 y Ch2 de la muestra 0
    """
    os.makedirs(output_dir, exist_ok=True)
    n_s, n_t, n_c = matriz_3D.shape

    # aplanar (n_s * n_t, n_c)
    data = matriz_3D.reshape(-1, n_c)
    cols = [f"Ch{i+1}" for i in range(n_c)]
    df = pd.DataFrame(data, columns=cols)

    # 1) Estadísticos descriptivos
    stats = df.describe().transpose()
    stats.to_csv(os.path.join(output_dir, "descriptive_stats_per_channel.csv"))
    print("    – Estadísticos descriptivos guardados")

    # 2) Histogramas
    df.hist(bins=30, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograms_channels.png"))
    plt.close('all')
    print("    – Histograms guardados")

    # 2.2) Boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot.box(ax=ax)
    ax.set_title("Boxplot por canal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_channels.png"))
    plt.close(fig)
    print("    – Boxplot guardado")

    # 3) Matriz de correlación + heatmap
    corr = df.corr()
    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap de correlación entre canales")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corr_heatmap.png"))
    plt.close(fig)
    print("    – Heatmap de correlación guardado")

    # 4) Scatter Ch1 vs Ch2 (muestra aleatoria)
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x="Ch1", y="Ch2", data=sample, alpha=0.3, ax=ax)
    ax.set_title("Scatter Ch1 vs Ch2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_Ch1_Ch2.png"))
    plt.close(fig)
    print("    – Scatter Ch1 vs Ch2 guardado")

    # 5) Cross-correlation Ch1 vs Ch2 de la muestra 0
    ts1 = matriz_3D[0, :, 0]
    ts2 = matriz_3D[0, :, 1]
    x1 = (ts1 - ts1.mean()) / ts1.std()
    x2 = (ts2 - ts2.mean()) / ts2.std()
    cc = np.correlate(x1, x2, mode="full")
    lags = np.arange(-len(ts1) + 1, len(ts1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, cc)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-correlation Ch1 vs Ch2 (muestra 0)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_correlation_Ch1_Ch2.png"))
    plt.close(fig)
    print("    – Cross-correlation guardada")

if __name__ == "__main__":
    base_input      = "severidad_alta"
    resultados_root = os.path.join(base_input, "resultados")
    skip = {"resultados", "resultados_finales"}

    for clase in sorted(os.listdir(base_input)):
        path_clase = os.path.join(base_input, clase)
        if not os.path.isdir(path_clase) or clase in skip:
            continue

        print(f"\nProcesando clase: {clase}")
        # 1) Intentar leer HDF5 existente
        h5_path = os.path.join(resultados_root, clase, "matriz_3D.h5")
        if os.path.isfile(h5_path):
            with h5py.File(h5_path, "r") as f:
                matriz = f["matriz_3D"][()]
            print(f"  • Cargada matriz 3D desde HDF5: {h5_path}")
        else:
            # 2) Si no existe, reconstruir desde CSV base
            matriz = cargar_y_formar_matriz(path_clase)

        # 3) Generar EDA en resultados/<clase>/eda/
        out_eda = os.path.join(resultados_root, clase, "eda")
        eda_relations(matriz, out_eda)

