import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# =====================================================
# 1. Lectura de datos (4 CSV) + Normalización
# =====================================================
def load_data():
    """
    Lee 4 CSV (align parallel 1..4), cada uno con un grado de severidad distinto.
    Se asume que cada CSV está 'aplanado': N filas x 1800 columnas (300*6).
    Se hace reshape a (N,300,6).
    """

    # Rutas a tus CSV (ajusta según tu carpeta)
    path_1 = os.path.join("resultados", "align parallel 1", "matriz_3D_aplanada.csv")
    path_2 = os.path.join("resultados", "align parallel 2", "matriz_3D_aplanada.csv")
    path_3 = os.path.join("resultados", "align parallel 3", "matriz_3D_aplanada.csv")
    path_4 = os.path.join("resultados", "align parallel 4", "matriz_3D_aplanada.csv")

    df1 = pd.read_csv(path_1, header=0)
    df2 = pd.read_csv(path_2, header=0)
    df3 = pd.read_csv(path_3, header=0)
    df4 = pd.read_csv(path_4, header=0)

    # Convierte a numpy
    X_data_1 = df1.values  # (N1, 1800)
    X_data_2 = df2.values
    X_data_3 = df3.values
    X_data_4 = df4.values

    n_instancias = 300
    n_canales = 6

    # Reshape a (num_samples, 300, 6)
    X_data_1 = X_data_1.reshape((-1, n_instancias, n_canales))
    X_data_2 = X_data_2.reshape((-1, n_instancias, n_canales))
    X_data_3 = X_data_3.reshape((-1, n_instancias, n_canales))
    X_data_4 = X_data_4.reshape((-1, n_instancias, n_canales))

    # Etiquetas
    y_data_1 = np.full((X_data_1.shape[0],), 0, dtype=int)
    y_data_2 = np.full((X_data_2.shape[0],), 1, dtype=int)
    y_data_3 = np.full((X_data_3.shape[0],), 2, dtype=int)
    y_data_4 = np.full((X_data_4.shape[0],), 3, dtype=int)

    X = np.concatenate([X_data_1, X_data_2, X_data_3, X_data_4], axis=0)
    y = np.concatenate([y_data_1, y_data_2, y_data_3, y_data_4], axis=0)

    print("Shape de X (después de reshape):", X.shape)
    print("Shape de y:", y.shape)

    # === (Opcional) Normalización min–max global ===
    # Calculamos min y max en TODOS los datos y TODOS los canales
    X_min = X.min(axis=(0,1), keepdims=True)  # shape (1,1,6)
    X_max = X.max(axis=(0,1), keepdims=True)  # shape (1,1,6)
    eps = 1e-8
    X = (X - X_min) / (X_max - X_min + eps)

    return X, y

# =====================================================
# 2. Modelo CNN "más grande"
# =====================================================
def build_model(n_instancias, n_canales):
    """
    - Tres capas Conv1D con 128 filtros y kernel_size distinto.
    - Dropout alto (0.5) para regularizar.
    - Dense(256) final.
    """
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
# 3. Saliency Map
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
    """
    - X_signal: shape (300, 6)
    - saliency: shape (300, 6)
    - save_path: ruta para guardar el PNG
    """
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
    plt.close(fig)  # cerramos para no saturar el entorno

# =====================================================
# 4. Script Principal
# =====================================================
if __name__ == "__main__":
    # Creamos carpeta para guardar resultados
    os.makedirs("resultados_finales", exist_ok=True)
    # Subcarpeta para saliency
    saliency_dir = os.path.join("resultados_finales", "saliency")
    os.makedirs(saliency_dir, exist_ok=True)

    # 4.1 Cargamos datos
    X, y = load_data()
    n_instancias = X.shape[1]
    n_canales = X.shape[2]

    # 4.2 Validación Cruzada
    from sklearn.model_selection import KFold
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    histories = []

    # Abrimos un archivo para log de folds
    log_folds_path = os.path.join("resultados_finales", "resultados_folds.txt")
    with open(log_folds_path, "w") as f_log:
        f_log.write("=== Validación Cruzada (K-Fold) ===\n")

        for train_index, val_index in kf.split(X):
            f_log.write(f"\n--- Fold {fold_no} ---\n")
            print(f"\n--- Fold {fold_no} ---")

            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            
            model = build_model(n_instancias, n_canales)
            history = model.fit(
                X_train_cv, y_train_cv,
                epochs=50,        # más épocas
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

        # Promedio
        mean_acc = np.mean(acc_per_fold)
        std_acc = np.std(acc_per_fold)
        mean_loss = np.mean(loss_per_fold)

        f_log.write("\nPromedio de todos los folds:\n")
        f_log.write(f"> Accuracy: {mean_acc*100:.2f}% (± {std_acc*100:.2f}%)\n")
        f_log.write(f"> Loss: {mean_loss:.4f}\n")

        print("\nPromedio de todos los folds:")
        print(f"> Accuracy: {mean_acc*100:.2f}% (± {std_acc*100:.2f}%)")
        print(f"> Loss: {mean_loss:.4f}")

    # Graficamos la curva de entrenamiento del primer fold y la guardamos
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

    # 4.3 Entrenamiento final con TODOS los datos
    model_final = build_model(n_instancias, n_canales)
    history_final = model_final.fit(X, y, epochs=50, batch_size=32, verbose=0)

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

    # 4.4 Evaluación con Ruido (0.05)
    noise_factor = 0.05
    X_augmented = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    # Si crees que tus datos deben permanecer en [0,1], haz clip:
    # X_augmented = np.clip(X_augmented, 0, 1)

    scores_aug = model_final.evaluate(X_augmented, y, verbose=0)
    aug_loss = scores_aug[0]
    aug_acc = scores_aug[1]

    # Predecimos
    y_pred = np.argmax(model_final.predict(X_augmented), axis=1)
    classif_report = classification_report(y, y_pred)
    conf_mat = confusion_matrix(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)

    # Guardamos en un archivo
    eval_aug_path = os.path.join("resultados_finales", "eval_ruido_0.05.txt")
    with open(eval_aug_path, "w") as f_eval:
        f_eval.write(f"=== Evaluación en Datos Aumentados (ruido={noise_factor}) ===\n")
        f_eval.write(f"Loss: {aug_loss:.4f} | Accuracy: {aug_acc*100:.2f}%\n\n")
        f_eval.write("Reporte de clasificación:\n")
        f_eval.write(classif_report + "\n")
        f_eval.write("Matriz de confusión:\n")
        f_eval.write(str(conf_mat) + "\n")
        f_eval.write(f"Balanced Accuracy: {bal_acc:.2f}\n")

    # 4.5 Mapas de Saliencia (ejemplos de cada clase)
    classes = [0,1,2,3]
    for c in classes:
        indices_c = np.where(y == c)[0]
        if len(indices_c) == 0:
            continue
        # Elegimos la primera muestra
        idx = indices_c[0]
        X_sample_c = X[idx:idx+1]  # (1, 300, 6)
        sal_map_c = compute_saliency_map(model_final, X_sample_c)
        
        # Guardamos una sola figura con 6 subplots (1 por canal)
        save_fig_path = os.path.join(saliency_dir, f"saliency_clase_{c}.png")
        plot_saliency_map(
            X_sample_c[0],
            sal_map_c[0],
            titulo=f"Mapa de Saliencia (Clase {c})",
            save_path=save_fig_path
        )

    # 4.6 Diferentes niveles de ruido y predicción
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    eval_ruidos_path = os.path.join("resultados_finales", "eval_diferentes_ruidos.txt")
    with open(eval_ruidos_path, "w") as f_ruidos:
        f_ruidos.write("=== Evaluación con diferentes niveles de ruido ===\n")
        for nl in noise_levels:
            X_noisy = X + nl * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
            # X_noisy = np.clip(X_noisy, 0, 1)
            
            scores_noisy = model_final.evaluate(X_noisy, y, verbose=0)
            noisy_loss = scores_noisy[0]
            noisy_acc = scores_noisy[1]
            f_ruidos.write(f"Ruido={nl} -> Loss: {noisy_loss:.4f} | Accuracy: {noisy_acc*100:.2f}%\n")
