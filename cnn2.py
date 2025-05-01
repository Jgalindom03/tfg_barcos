import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# =====================================================
# 1. Lectura de datos desde signals.csv y labels.csv
# =====================================================
def load_data(data_dir="data"):
    signals_path = os.path.join(data_dir, "signals.csv")
    labels_path  = os.path.join(data_dir, "labels.csv")
    
    df_signals = pd.read_csv(signals_path, header=None)
    df_labels  = pd.read_csv(labels_path, header=None)
    
    return df_signals.values, df_labels.values

# =====================================================
# 2. Modelo CNN (Conv1D)
# =====================================================
def build_model(input_length, num_classes):
    """
    Modelo CNN1D mejorado para reducir sobreajuste.
    """
    #l2_reg = keras.regularizers.l2(1e-4)

    model = keras.Sequential([
        keras.Input(shape=(input_length, 1)),

        layers.Conv1D(filters=16, kernel_size=3, padding='same'),
        #layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=1),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

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
    Calcula el mapa de saliencia para la muestra dada.
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
# 4. Script Principal con separación de Test
# =====================================================
if __name__ == "__main__":
    # Directorio de resultados dentro de la carpeta data
    results_dir = os.path.join("data", "resultados_finales")
    os.makedirs(results_dir, exist_ok=True)

    saliency_dir = os.path.join(results_dir, "saliency")
    os.makedirs(saliency_dir, exist_ok=True)

    # 4.1 Cargamos datos
    X, y = load_data(data_dir="data")
    y = y - 1

    transformations = []
    for i in range(X.shape[0]):
        transformations.append(MinMaxScaler().fit_transform(X[i, :].reshape(-1, 1)).flatten())
    X = transformations
    X = np.array(X)  # Convertir a numpy array
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Añadimos dimensión de canal


    print(X.shape, y.shape)

    # (Opcional) Graficar algunas señales para verificar y guardarlas
    for i in range(4):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(X[i, :, 0], label=f"Label: {y[i]+1}")
        ax.set_title(f"Señal {i} - Etiqueta: {y[i]+1}")
        ax.set_xlabel("Instancias")
        ax.set_ylabel("Potencia Normalizada")
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(results_dir, f"senal_{i}_etiqueta_{y[i]+1}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Guardado: {save_path}")


    num_clases = len(np.unique(y))
    input_length = X.shape[1]

    # 4.2 Separación en conjuntos de entrenamiento y test
    # Se utiliza 80% para entrenamiento y 20% para test, manteniendo la estratificación de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Shape de X_train:", X_train.shape)
    print("Shape de X_test:", X_test.shape)

    y_train = y_train.reshape(1, -1)[0]  # Convertir a 1D
    y_test = y_test.reshape(1, -1)[0]    # Convertir a 1D

    print(np.unique(y))
    
    # 4.5 Entrenamiento final con TODOS los datos de entrenamiento y Early Stopping
    model_final = build_model(input_length, num_classes=num_clases)
    early_stop_final = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history_final = model_final.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop_final],
        verbose=1
    )

    fig_final, axf = plt.subplots(1, 2, figsize=(12,5))
    axf[0].plot(history_final.history['loss'], label='Train Loss')
    axf[0].set_title('Curva de Loss (Final - Entrenamiento)')
    axf[0].set_xlabel('Epoch')
    axf[0].set_ylabel('Loss')
    axf[0].legend()

    axf[1].plot(history_final.history['accuracy'], label='Train Accuracy')
    axf[1].set_title('Curva de Accuracy (Final - Entrenamiento)')
    axf[1].set_xlabel('Epoch')
    axf[1].set_ylabel('Accuracy')
    axf[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "curvas_final.png"), dpi=150)
    plt.close(fig_final)

    # 4.6 Evaluación final del modelo en el conjunto de Test
    scores_test = model_final.evaluate(X_test, y_test, verbose=0)
    print(f"Test -> Loss: {scores_test[0]:.4f} | Accuracy: {scores_test[1]*100:.2f}%")
    with open(os.path.join(results_dir, "resultados_test.txt"), "w") as f_test:
        f_test.write(f"Test -> Loss: {scores_test[0]:.4f} | Accuracy: {scores_test[1]*100:.2f}%\n")

    # 4.7 Evaluación en Test con diferentes niveles de ruido y reporte de clasificación
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    eval_ruidos_path = os.path.join(results_dir, "eval_diferentes_ruidos_test.txt")
    with open(eval_ruidos_path, "w") as f_ruidos:
        f_ruidos.write("=== Evaluación en Test con diferentes niveles de ruido ===\n")
        for nl in noise_levels:
            X_test_noisy = X_test + nl * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
            scores_noisy = model_final.evaluate(X_test_noisy, y_test, verbose=0)
            noisy_loss = scores_noisy[0]
            noisy_acc = scores_noisy[1]
            f_ruidos.write(f"\nRuido={nl} -> Loss: {noisy_loss:.4f} | Accuracy: {noisy_acc*100:.2f}%\n")
            
            # Predicciones y reporte de clasificación en Test
            y_pred = np.argmax(model_final.predict(X_test_noisy), axis=1)
            report = classification_report(y_test, y_pred)
            f_ruidos.write("Reporte de clasificación:\n")
            f_ruidos.write(report + "\n")
            
