import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_planos(resultados_root):
    """
    Lee todos los CSV 'matriz_3D_aplanada_dataframe.csv' en
    resultados_root/<clase>/ y devuelve un único DataFrame con
    una columna 'clase'.
    """
    dfs = []
    for clase in sorted(os.listdir(resultados_root)):
        carpeta = os.path.join(resultados_root, clase)
        ruta_csv = os.path.join(carpeta, "matriz_3D_aplanada_dataframe.csv")
        if os.path.isfile(ruta_csv):
            df = pd.read_csv(ruta_csv)
            df["clase"] = clase
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No se encontró ningún CSV en {resultados_root}")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Cargados {len(dfs)} clases → DataFrame con {df_all.shape[0]} filas y {df_all.shape[1]} columnas")
    return df_all

def eda_comparativo(df_all, output_dir):
    """
    Genera un EDA comparativo entre clases:
      1) Estadísticos descriptivos por clase
      2) Boxplots por canal+clase
      3) Histograms facetados
      4) Scatter Ch1 vs Ch2 coloreado por clase
      5) Matrices de correlación por clase
    """
    os.makedirs(output_dir, exist_ok=True)
    canales = [c for c in df_all.columns if c.startswith("Ch")]

    # 1) Descriptivos por clase
    stats = df_all.groupby("clase")[canales].describe().transpose()
    stats.to_csv(os.path.join(output_dir, "descriptive_stats_por_clase.csv"))
    print("  • Descriptivos por clase guardados")

    # 2) Boxplots por canal
    for ch in canales:
        plt.figure(figsize=(8,4))
        sns.boxplot(x="clase", y=ch, data=df_all)
        plt.title(f"Boxplot {ch} por clase")
        plt.xticks(rotation=30)
        plt.tight_layout()
        fn = f"boxplot_{ch}_por_clase.png"
        plt.savefig(os.path.join(output_dir, fn))
        plt.close()
    print("  • Boxplots por canal guardados")

    # 3) Histograms facetados (ejemplo con Ch1)
    g = sns.FacetGrid(df_all, col="clase", col_wrap=3, sharex=False, sharey=False)
    g.map(plt.hist, "Ch1", bins=30)
    g.fig.suptitle("Histogramas de Ch1 por clase", y=1.02)
    g.tight_layout()
    g.savefig(os.path.join(output_dir, "histograms_Ch1_por_clase.png"))
    plt.close()
    print("  • Histograms facetados guardados")

    # 4) Scatter Ch1 vs Ch2 coloreado por clase
    sample = df_all.sample(n=min(5000, len(df_all)), random_state=0)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x="Ch1", y="Ch2", hue="clase", data=sample, alpha=0.4)
    plt.title("Ch1 vs Ch2 por clase")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_Ch1_Ch2_por_clase.png"))
    plt.close()
    print("  • Scatter Ch1 vs Ch2 guardado")

    # 5) Correlaciones por clase
    for clase, grp in df_all.groupby("clase"):
        corr = grp[canales].corr()
        plt.figure(figsize=(6,5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Correlación canales — {clase}")
        plt.tight_layout()
        fn = f"corr_heatmap_{clase.replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, fn))
        plt.close()
    print("  • Correlaciones por clase guardadas")

if __name__ == "__main__":
    base_input      = "cavitation suction"
    resultados_root = os.path.join(base_input, "resultados")
    out_comparativo = os.path.join(resultados_root, "EDA_comparativo")

    # 1) Cargar todo
    df_all = cargar_planos(resultados_root)

    # 2) Hacer EDA comparativo
    eda_comparativo(df_all, out_comparativo)
