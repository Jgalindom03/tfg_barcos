import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# 1. Lectura de datos (4 CSV) + troceado en ventanas
# =====================================================
def load_data_aplanado():
    """
    Lee 4 CSV (align parallel 1..4), cada uno con un grado de severidad distinto.
    Se asume que cada CSV tiene N filas x 1800 columnas (300*6) en forma 'aplanada'.
    Cada fila es una muestra.
    """
    path_1 = os.path.join("resultados", "align parallel 1", "matriz_3D_aplanada.csv")
    path_2 = os.path.join("resultados", "align parallel 2", "matriz_3D_aplanada.csv")
    path_3 = os.path.join("resultados", "align parallel 3", "matriz_3D_aplanada.csv")
    path_4 = os.path.join("resultados", "align parallel 4", "matriz_3D_aplanada.csv")


    df1 = pd.read_csv(path_1, header=0)
    df2 = pd.read_csv(path_2, header=0)
    df3 = pd.read_csv(path_3, header=0)
    df4 = pd.read_csv(path_4, header=0)

    X_data_1 = df1.values  # (N1, 1800)
    X_data_2 = df2.values
    X_data_3 = df3.values
    X_data_4 = df4.values

    y_data_1 = np.full((X_data_1.shape[0],), 0, dtype=int)
    y_data_2 = np.full((X_data_2.shape[0],), 1, dtype=int)
    y_data_3 = np.full((X_data_3.shape[0],), 2, dtype=int)
    y_data_4 = np.full((X_data_4.shape[0],), 3, dtype=int)

    X_apl = np.concatenate([X_data_1, X_data_2, X_data_3, X_data_4], axis=0)
    y = np.concatenate([y_data_1, y_data_2, y_data_3, y_data_4], axis=0)

    print("Forma de X_apl (aplanado):", X_apl.shape)
    print("Forma de y:", y.shape)

    return X_apl, y

def load_data_ventanas():
    """
    Lee 4 CSV (cada uno con 210000 filas y 6 columnas) y trocea en ventanas de 300.
    Esto es el caso cuando cada CSV = (210000,6).
    Se obtienen (700,300,6) por CSV.
    """
    path_1 = os.path.join("resultados", "align parallel 1", "matriz_3D_aplanada.csv")
    path_2 = os.path.join("resultados", "align parallel 2", "matriz_3D_aplanada.csv")
    path_3 = os.path.join("resultados", "align parallel 3", "matriz_3D_aplanada.csv")
    path_4 = os.path.join("resultados", "align parallel 4", "matriz_3D_aplanada.csv")


    df1 = pd.read_csv(path_1, header=0)
    df2 = pd.read_csv(path_2, header=0)
    df3 = pd.read_csv(path_3, header=0)
    df4 = pd.read_csv(path_4, header=0)

    # asumiendo que cada df = (210000,6)
    arr1 = df1.values
    arr2 = df2.values
    arr3 = df3.values
    arr4 = df4.values

    # troceamos en (700,300,6)
    def reshape_ventanas(arr, label):
        n_total = (arr.shape[0] // 300) * 300
        arr = arr[:n_total]  # descartar filas sobrantes
        num_samples = n_total // 300
        X_data = arr.reshape(num_samples, 300, 6)
        y_data = np.full((num_samples,), label, dtype=int)
        return X_data, y_data

    X_data_1, y_data_1 = reshape_ventanas(arr1, 0)
    X_data_2, y_data_2 = reshape_ventanas(arr2, 1)
    X_data_3, y_data_3 = reshape_ventanas(arr3, 2)
    X_data_4, y_data_4 = reshape_ventanas(arr4, 3)

    X = np.concatenate([X_data_1, X_data_2, X_data_3, X_data_4], axis=0)
    y = np.concatenate([y_data_1, y_data_2, y_data_3, y_data_4], axis=0)

    print("Forma de X (ventanas):", X.shape)  # (2800,300,6) si 700 x 4
    print("Forma de y:", y.shape)
    return X, y

# =====================================================
# 2. Ejemplo de extracción de características
# =====================================================
def extract_features(X_3d, method="fft"):
    """
    X_3d: (N, 300, 6)
    method: "fft" para extraer magnitud del espectro (frecuencia),
            "stats" para estadísticos básicos,
            etc.
    Devuelve X_feat (N, num_features).
    """
    N, n_inst, n_canales = X_3d.shape

    if method == "fft":
        # Calculamos la FFT en cada ventana y cada canal
        # (Podríamos quedarnos con las primeras 150 frecuencias, p.ej.)
        fft_size = n_inst // 2  # ej. 150 si 300
        X_feat = []
        for i in range(N):
            window = X_3d[i]  # shape (300,6)
            # Calculamos FFT para cada canal
            feats_window = []
            for c in range(n_canales):
                sig = window[:, c]
                fft_vals = np.fft.rfft(sig)  # tamaño ~151 si 300
                mag = np.abs(fft_vals)
                # Te quedas con todo, o recortas
                mag = mag[:fft_size]  # 150
                feats_window.append(mag)
            feats_window = np.concatenate(feats_window, axis=0)  # (6*150,)
            X_feat.append(feats_window)
        X_feat = np.array(X_feat)  # (N, 6*150=900)
        return X_feat

    elif method == "stats":
        # Calculamos media, std, max, min, RMS, etc. para cada canal
        # Ejemplo sencillo:
        X_feat = []
        for i in range(N):
            window = X_3d[i]  # (300,6)
            feats_window = []
            for c in range(n_canales):
                sig = window[:, c]
                mean_ = np.mean(sig)
                std_ = np.std(sig)
                maxi_ = np.max(sig)
                mini_ = np.min(sig)
                # RMS
                rms_ = np.sqrt(np.mean(sig**2))
                feats_window += [mean_, std_, maxi_, mini_, rms_]
            X_feat.append(feats_window)
        X_feat = np.array(X_feat)  # shape (N, 5*n_canales)
        return X_feat

    else:
        raise ValueError("Método de extracción de features no reconocido.")

# =====================================================
# 3. RandomForest (u otro clasificador tradicional)
# =====================================================
def test_random_forest(X_3d, y):
    """
    Demuestra cómo usar un clasificador tradicional (RandomForest).
    - O bien usas X_3d crudo aplanado: (N, 300*6)
    - O aplicas extract_features(X_3d) antes.
    """
    # Aplanar crudo (N,1800) [OPCIONAL]
    N, n_inst, n_canales = X_3d.shape
    X_flat = X_3d.reshape(N, n_inst*n_canales)

    # O usar un feature extraction:
    # X_feat = extract_features(X_3d, method="fft")
    # X_feat = extract_features(X_3d, method="stats")

    # Aquí, usaremos X_flat crudo para la demo
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RandomForest accuracy (crudo aplanado): {acc*100:.2f}%")

# =====================================================
# 4. CNN
# =====================================================
def build_model(n_instancias, n_canales):
    model = keras.Sequential([
        keras.Input(shape=(n_instancias, n_canales)),
        layers.Conv1D(filters=128, kernel_size=7, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 clases
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =====================================================
# 5. Saliency Map
# =====================================================
def compute_saliency_map(model, X_sample, class_idx=None):
    X_var = tf.Variable(X_sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_var)
        predictions = model(X_var)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[0, class_idx]
    grads = tape.gradient(loss, X_var)
    saliency = tf.math.abs(grads)
    return saliency.numpy()

def plot_saliency_map(X_signal, saliency, titulo="Mapa de Saliencia", save_path=None):
    n_canales = X_signal.shape[1]
    fig, axes = plt.subplots(n_canales, 1, figsize=(12, 3*n_canales))
    
    if n_canales == 1:
        axes = [axes]
    
    for i in range(n_canales):
        axes[i].plot(X_signal[:, i], label='Señal')
        axes[i].plot(saliency[:, i], label='Saliencia')
        axes[i].set_title(f'Canal {i}')
        axes[i].legend()
    
    plt.suptitle(titulo)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =====================================================
# 6. Script Principal
# =====================================================
if __name__ == "__main__":
    # -----------------------------------------------------
    # OPCIÓN A: Cargar datos aplanados (N,1800) -> (N,300,6)
    # (si cada CSV ya es 1800 columnas)
    # -----------------------------------------------------
    # X_apl, y = load_data_aplanado()
    # # Damos forma 3D si no la tuviera
    # # Suponiendo que X_apl.shape[1] == 1800
    # N = X_apl.shape[0]
    # X_3d = X_apl.reshape(N, 300, 6)

    # -----------------------------------------------------
    # OPCIÓN B: Cargar datos con 210000 filas y 6 col -> trocear a (700,300,6)
    # -----------------------------------------------------
    X_3d, y = load_data_ventanas()

    print("Shape final X_3d:", X_3d.shape)
    print("Shape y:", y.shape)

    # Normalización min-max global
    X_min = X_3d.min(axis=(0,1), keepdims=True)
    X_max = X_3d.max(axis=(0,1), keepdims=True)
    eps = 1e-8
    X_3d = (X_3d - X_min) / (X_max - X_min + eps)

    # -----------------------------------------------------
    # 6.1 Prueba con RandomForest (clásico) para ver si supera 25%
    # -----------------------------------------------------
    test_random_forest(X_3d, y)

    # -----------------------------------------------------
    # 6.2 CNN con K-Fold
    # -----------------------------------------------------
    os.makedirs("resultados_finales", exist_ok=True)
    saliency_dir = os.path.join("resultados_finales", "saliency")
    os.makedirs(saliency_dir, exist_ok=True)

    n_instancias = X_3d.shape[1]  # 300
    n_canales = X_3d.shape[2]     # 6

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    histories = []

    log_folds_path = os.path.join("resultados_finales", "resultados_folds.txt")
    with open(log_folds_path, "w") as f_log:
        f_log.write("=== Validación Cruzada (K-Fold) ===\n")

        for train_index, val_index in kf.split(X_3d):
            f_log.write(f"\n--- Fold {fold_no} ---\n")
            print(f"\n--- Fold {fold_no} ---")

            X_train_cv, X_val_cv = X_3d[train_index], X_3d[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            
            model = build_model(n_instancias, n_canales)
            history = model.fit(
                X_train_cv, y_train_cv,
                epochs=50,
                batch_size=32,
                validation_data=(X_val_cv, y_val_cv),
                verbose=0
            )
            
            scores = model.evaluate(X_val_cv, y_val_cv, verbose=0)
            fold_loss = scores[0]
            fold_acc = scores[1]
            print(f"Fold {fold_no} -> Loss: {fold_loss:.4f} | Accuracy: {fold_acc*100:.2f}%")
            f_log.write(f"Fold {fold_no} -> Loss: {fold_loss:.4f} | Accuracy: {fold_acc*100:.2f}%\n")

            acc_per_fold.append(fold_acc)
            loss_per_fold.append(fold_loss)
            histories.append(history)
            fold_no += 1

        mean_acc = np.mean(acc_per_fold)
        std_acc = np.std(acc_per_fold)
        mean_loss = np.mean(loss_per_fold)

        f_log.write("\nPromedio de todos los folds:\n")
        f_log.write(f"> Accuracy: {mean_acc*100:.2f}% (± {std_acc*100:.2f}%)\n")
        f_log.write(f"> Loss: {mean_loss:.4f}\n")

        print("\nPromedio de todos los folds:")
        print(f"> Accuracy: {mean_acc*100:.2f}% (± {std_acc*100:.2f}%)")
        print(f"> Loss: {mean_loss:.4f}")

    # Curva de entrenamiento del primer fold
    history_example = histories[0]
    fig_curvas, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(history_example.history['loss'], label='Train Loss')
    ax[0].plot(history_example.history['val_loss'], label='Val Loss')
    ax[0].set_title('Curva de Loss (Fold 1)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(history_example.history['accuracy'], label='Train Accuracy')
    ax[1].plot(history_example.history['val_accuracy'], label='Val Accuracy')
    ax[1].set_title('Curva de Accuracy (Fold 1)')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("resultados_finales", "curvas_fold_1.png"), dpi=150)
    plt.close(fig_curvas)

    # Entrenamiento final con TODOS los datos
    model_final = build_model(n_instancias, n_canales)
    history_final = model_final.fit(X_3d, y, epochs=50, batch_size=32, verbose=0)

    # Guardamos curva de entrenamiento final
    fig_final, axf = plt.subplots(1, 2, figsize=(12,5))
    axf[0].plot(history_final.history['loss'], label='Train Loss')
    axf[0].set_title('Curva de Loss (Final)')
    axf[0].set_xlabel('Epoch')
    axf[0].set_ylabel('Loss')
    axf[0].legend()

    axf[1].plot(history_final.history['accuracy'], label='Train Accuracy')
    axf[1].set_title('Curva de Accuracy (Final)')
    axf[1].set_xlabel('Epoch')
    axf[1].set_ylabel('Accuracy')
    axf[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("resultados_finales", "curvas_final.png"), dpi=150)
    plt.close(fig_final)

    # Evaluación con Ruido (0.05)
    noise_factor = 0.05
    X_augmented = X_3d + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_3d.shape)
    scores_aug = model_final.evaluate(X_augmented, y, verbose=0)
    aug_loss = scores_aug[0]
    aug_acc = scores_aug[1]

    y_pred = np.argmax(model_final.predict(X_augmented), axis=1)
    classif_report = classification_report(y, y_pred)
    conf_mat = confusion_matrix(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)

    eval_aug_path = os.path.join("resultados_finales", "eval_ruido_0.05.txt")
    with open(eval_aug_path, "w") as f_eval:
        f_eval.write(f"=== Evaluación en Datos Aumentados (ruido={noise_factor}) ===\n")
        f_eval.write(f"Loss: {aug_loss:.4f} | Accuracy: {aug_acc*100:.2f}%\n\n")
        f_eval.write("Reporte de clasificación:\n")
        f_eval.write(classif_report + "\n")
        f_eval.write("Matriz de confusión:\n")
        f_eval.write(str(conf_mat) + "\n")
        f_eval.write(f"Balanced Accuracy: {bal_acc:.2f}\n")

    # Mapas de Saliencia
    classes = [0,1,2,3]
    for c in classes:
        indices_c = np.where(y == c)[0]
        if len(indices_c) == 0:
            continue
        idx = indices_c[0]
        X_sample_c = X_3d[idx:idx+1]
        sal_map_c = compute_saliency_map(model_final, X_sample_c)
        save_fig_path = os.path.join(saliency_dir, f"saliency_clase_{c}.png")
        plot_saliency_map(
            X_sample_c[0],
            sal_map_c[0],
            titulo=f"Mapa de Saliencia (Clase {c})",
            save_path=save_fig_path
        )

    # Diferentes niveles de ruido
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    eval_ruidos_path = os.path.join("resultados_finales", "eval_diferentes_ruidos.txt")
    with open(eval_ruidos_path, "w") as f_ruidos:
        f_ruidos.write("=== Evaluación con diferentes niveles de ruido ===\n")
        for nl in noise_levels:
            X_noisy = X_3d + nl * np.random.normal(loc=0.0, scale=1.0, size=X_3d.shape)
            scores_noisy = model_final.evaluate(X_noisy, y, verbose=0)
            noisy_loss = scores_noisy[0]
            noisy_acc = scores_noisy[1]
            f_ruidos.write(f"Ruido={nl} -> Loss: {noisy_loss:.4f} | Accuracy: {noisy_acc*100:.2f}%\n")
