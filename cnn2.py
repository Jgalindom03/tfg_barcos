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
# 1. Lectura de datos con padding para columnas variables
# =====================================================
def load_data(base_dir="severidad_alta/resultados"):
    """
    Lee todas las subcarpetas dentro de 'base_dir'.
    - En cada subcarpeta busca el archivo 'matriz_3D_aplanada.csv'.
    - Se realiza padding hasta el máximo número de columnas encontrado.
    
    Retorna:
      X : np.ndarray, shape (N_total, max_cols)
      y : np.ndarray, shape (N_total,)
    """
    subcarpetas = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    data_list = []
    label_list = []
    current_label = 0
    max_cols = 0

    for carpeta in subcarpetas:
        csv_path = os.path.join(base_dir, carpeta, "matriz_3D_aplanada.csv")
        if not os.path.isfile(csv_path):
            print(f"[Aviso] No se encontró {csv_path}. Se omite la carpeta '{carpeta}'.")
            continue

        print(f"Leyendo: {csv_path} (Etiqueta = {current_label})")
        df = pd.read_csv(csv_path, header=0)
        arr = df.values

        data_list.append(arr)
        label_list.append(np.full((arr.shape[0],), current_label, dtype=int))

        if arr.shape[1] > max_cols:
            max_cols = arr.shape[1]

        current_label += 1

    if not data_list:
        raise ValueError(f"No se encontraron CSV válidos en '{base_dir}'.")

    X_list = []
    y_list = []
    for arr, labels in zip(data_list, label_list):
        N, M_i = arr.shape
        arr_padded = np.zeros((N, max_cols), dtype=arr.dtype)
        arr_padded[:, :M_i] = arr
        X_list.append(arr_padded)
        y_list.append(labels)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print("Shape final de X:", X.shape)
    print("Shape final de y:", y.shape)

    # Normalización min–max
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    eps = 1e-8
    X = (X - X_min) / (X_max - X_min + eps)

    return X, y

# =====================================================
# 2. Modelo CNN (con padding="same")
# =====================================================
def build_model(input_length, num_classes):
    """
    Construye un modelo CNN1D para entrada de forma (input_length, 1).
    """
    model = keras.Sequential([
        keras.Input(shape=(input_length, 1)),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
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
    """
    X_sample: tf.Variable con forma (1, input_length, 1)
    """
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

def plot_saliency_map(saliency, titulo="Mapa de Saliencia", save_path=None):
    """
    Grafica la saliencia usando un diagrama de barras.
    - saliency: np.ndarray de forma (M, 1)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    x_axis = np.arange(saliency.shape[0])
    ax.bar(x_axis, saliency[:, 0], label='Saliencia')
    ax.set_xlabel("Índice de característica")
    ax.set_ylabel("Valor de saliencia")
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =====================================================
# 4. Script Principal
# =====================================================
if __name__ == "__main__":
    results_dir = os.path.join("severidad_alta", "resultados_finales")
    os.makedirs(results_dir, exist_ok=True)

    saliency_dir = os.path.join(results_dir, "saliency")
    os.makedirs(saliency_dir, exist_ok=True)

    # 4.1 Cargamos datos (con padding de columnas)
    X, y = load_data(base_dir="severidad_alta/resultados")

    # Añadimos dimensión de canal: (N, M) -> (N, M, 1)
    X = X[..., np.newaxis]
    print("Nuevo shape de X para Conv1D:", X.shape)

    num_clases = len(np.unique(y))
    input_length = X.shape[1]

    # 4.2 Validación Cruzada con Early Stopping
    k = 2
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    histories = []

    log_folds_path = os.path.join(results_dir, "resultados_folds.txt")
    with open(log_folds_path, "w") as f_log:
        f_log.write("=== Validación Cruzada (K-Fold) ===\n")

        for train_index, val_index in kf.split(X):
            f_log.write(f"\n--- Fold {fold_no} ---\n")
            print(f"\n--- Fold {fold_no} ---")

            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            
            model = build_model(input_length, num_classes=num_clases)
            
            # Callback de Early Stopping
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            
            history = model.fit(
                X_train_cv, y_train_cv,
                epochs=5,
                batch_size=32,
                validation_data=(X_val_cv, y_val_cv),
                callbacks=[early_stop],
                verbose=1
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

    # 4.3 Graficamos la curva de entrenamiento del primer fold
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
    plt.savefig(os.path.join(results_dir, "curvas_fold_1.png"), dpi=150)
    plt.close(fig_curvas)

    # 4.4 Entrenamiento final con TODOS los datos y Early Stopping
    model_final = build_model(input_length, num_classes=num_clases)
    early_stop_final = keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    history_final = model_final.fit(X, y, epochs=10, batch_size=32, callbacks=[early_stop_final], verbose=1)

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
    plt.savefig(os.path.join(results_dir, "curvas_final.png"), dpi=150)
    plt.close(fig_final)

    # 4.5 Evaluación con diferentes niveles de ruido y reporte de clasificación
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    eval_ruidos_path = os.path.join(results_dir, "eval_diferentes_ruidos.txt")
    with open(eval_ruidos_path, "w") as f_ruidos:
        f_ruidos.write("=== Evaluación con diferentes niveles de ruido ===\n")
        for nl in noise_levels:
            X_noisy = X + nl * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
            # X_noisy = np.clip(X_noisy, 0, 1)
            
            scores_noisy = model_final.evaluate(X_noisy, y, verbose=0)
            noisy_loss = scores_noisy[0]
            noisy_acc = scores_noisy[1]
            f_ruidos.write(f"\nRuido={nl} -> Loss: {noisy_loss:.4f} | Accuracy: {noisy_acc*100:.2f}%\n")
            
            # Predicciones y reporte de clasificación
            y_pred = np.argmax(model_final.predict(X_noisy), axis=1)
            report = classification_report(y, y_pred)
            f_ruidos.write("Reporte de clasificación:\n")
            f_ruidos.write(report + "\n")
    
    # 4.6 Mapas de Saliencia (un ejemplo de cada clase)
    classes = np.unique(y)
    for c in classes:
        indices_c = np.where(y == c)[0]
        if len(indices_c) == 0:
            continue
        idx = indices_c[0]
        X_sample_c = X[idx:idx+1]  # (1, M, 1)
        sal_map_c = compute_saliency_map(model_final, X_sample_c)

        save_fig_path = os.path.join(saliency_dir, f"saliency_clase_{c}.png")
        plot_saliency_map(
            sal_map_c[0],    # Se pasa solo la saliencia
            titulo=f"Mapa de Saliencia (Clase {c})",
            save_path=save_fig_path
        )
