import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# =====================================================
# 1. Lectura de datos con padding para columnas variables
# =====================================================
def load_data(base_dir="cavitation suction/resultados"):
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

    X_list, y_list = [], []
    for arr, labels in zip(data_list, label_list):
        N, M_i = arr.shape
        padded = np.zeros((N, max_cols), dtype=arr.dtype)
        padded[:, :M_i] = arr
        X_list.append(padded)
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
# 3. Saliency Map (barras, solo saliencia)
# =====================================================
def compute_saliency_map(model, X_sample, class_idx=None):
    X_var = tf.Variable(X_sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_var)
        preds = model(X_var)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[0, class_idx]
    grads = tape.gradient(loss, X_var)
    return tf.math.abs(grads).numpy()


def plot_saliency_map(saliency, titulo="Mapa de Saliencia", save_path=None):
    """
    Grafica la saliencia usando un diagrama de barras.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    indices = np.arange(saliency.shape[0])
    # Asume saliency de forma (M, 1) o (M,)
    vals = saliency[:, 0] if saliency.ndim == 2 else saliency
    ax.bar(indices, vals, label='Saliencia')
    ax.set_xlabel("Índice de característica")
    ax.set_ylabel("Valor de saliencia")
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =====================================================
# 4. Script Principal con split Train/Test
# =====================================================
if __name__ == "__main__":
    results_dir = os.path.join("cavitation suction", "resultados_finales")
    os.makedirs(results_dir, exist_ok=True)

    saliency_dir = os.path.join(results_dir, "saliency")
    os.makedirs(saliency_dir, exist_ok=True)

    # 4.1 Carga y preprocesado
    X, y = load_data(base_dir="cavitation suction/resultados")
    X = X[..., np.newaxis]
    num_clases = len(np.unique(y))
    input_length = X.shape[1]

    # 4.2 Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4.3 Entrenamiento con EarlyStopping
    model = build_model(input_length, num_clases)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Guardar curvas de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss entrenamiento')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy entrenamiento')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "curvas_entrenamiento.png"), dpi=150)
    plt.close(fig)

    # 4.4 Evaluación en Test
    scores_test = model.evaluate(X_test, y_test, verbose=0)
    with open(os.path.join(results_dir, "resultados_test.txt"), "w") as f:
        f.write(f"Test -> Loss: {scores_test[0]:.4f} | Accuracy: {scores_test[1]*100:.2f}%\n")

    # 4.5 Reporte en Test
    y_pred = np.argmax(model.predict(X_test), axis=1)
    with open(os.path.join(results_dir, "resultados_test.txt"), "a") as f:
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)) + "\n")

    # 4.6 Evaluación con ruido en Test
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    with open(os.path.join(results_dir, "eval_ruido_test.txt"), "w") as f_ruidos:
        f_ruidos.write("=== Evaluación Test con distintos ruidos ===\n")
        for nl in noise_levels:
            Xn = X_test + nl * np.random.normal(size=X_test.shape)
            scores_n = model.evaluate(Xn, y_test, verbose=0)
            y_pred_n = np.argmax(model.predict(Xn), axis=1)
            bal_acc = balanced_accuracy_score(y_test, y_pred_n)
            f_ruidos.write(f"\nRuido={nl} -> Loss: {scores_n[0]:.4f} | Acc: {scores_n[1]*100:.2f}% | BalAcc: {bal_acc:.4f}\n")
            f_ruidos.write(classification_report(y_test, y_pred_n) + "\n")

    # 4.7 Mapas de Saliencia (barras) en Test
    for c in np.unique(y_test):
        idx = np.where(y_test == c)[0][0]
        Xs = X_test[idx:idx+1]
        sal = compute_saliency_map(model, Xs)
        save_path = os.path.join(saliency_dir, f"saliency_clase_{c}.png")
        plot_saliency_map(sal[0], titulo=f"Saliencia Clase {c}", save_path=save_path)