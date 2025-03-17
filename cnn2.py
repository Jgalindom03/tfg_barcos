import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# =============================
# Simulación de datos (ejemplo)
# =============================
n_samples_per_class = 100  # Ajusta según tus datos
n_instancias = 300         # Número de instancias por muestra
n_canales = 6              # Número de canales

X = []
y = []
for label in range(4):
    # Se simulan datos aleatorios, con un pequeño offset según la clase
    X_class = np.random.rand(n_samples_per_class, n_instancias, n_canales) + label * 0.1
    X.append(X_class)
    y += [label] * n_samples_per_class

X = np.concatenate(X, axis=0)
y = np.array(y)

# ================================================
# Función para construir el modelo (con regularización)
# ================================================
def build_model():
    model = keras.Sequential()
    # Primera capa convolucional
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                            input_shape=(n_instancias, n_canales)))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Dropout para regularización
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    # Capa densa con dropout
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    # Capa de salida: 4 neuronas para 4 clases
    model.add(layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ===================================
# 1. Validación Cruzada con K-Fold
# ===================================
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_no = 1
acc_per_fold = []
loss_per_fold = []
histories = []

print("=== Validación Cruzada (K-Fold) ===")
for train_index, val_index in kf.split(X):
    print(f"\n--- Fold {fold_no} ---")
    X_train_cv, X_val_cv = X[train_index], X[val_index]
    y_train_cv, y_val_cv = y[train_index], y[val_index]
    
    model = build_model()
    history = model.fit(X_train_cv, y_train_cv, epochs=10, batch_size=16,
                        validation_data=(X_val_cv, y_val_cv), verbose=0)
    
    scores = model.evaluate(X_val_cv, y_val_cv, verbose=0)
    print(f"Fold {fold_no} -> Loss: {scores[0]:.4f} | Accuracy: {scores[1]*100:.2f}%")
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    histories.append(history)
    fold_no += 1

print("\nPromedio de todos los folds:")
print(f"> Accuracy: {np.mean(acc_per_fold)*100:.2f}% (± {np.std(acc_per_fold)*100:.2f}%)")
print(f"> Loss: {np.mean(loss_per_fold):.4f}")

# ============================================
# 2. Graficar Curvas de Aprendizaje (ejemplo Fold 1)
# ============================================
history = histories[0]  # Se toma el historial del primer fold
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Loss entrenamiento')
plt.plot(history.history['val_loss'], label='Loss validación')
plt.title('Curva de Loss (Fold 1)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Accuracy entrenamiento')
plt.plot(history.history['val_accuracy'], label='Accuracy validación')
plt.title('Curva de Accuracy (Fold 1)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ============================================
# 3. Evaluación con Datos Aumentados (Añadiendo Ruido)
# ============================================
# Entrenamos un modelo final con todos los datos
model_final = build_model()
history_final = model_final.fit(X, y, epochs=10, batch_size=16, verbose=0)

# Generamos datos aumentados añadiendo ruido gaussiano
noise_factor = 0.05
X_augmented = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
X_augmented = np.clip(X_augmented, 0.0, 1.0)  # Limitar a [0,1] en caso de imágenes o escalas similares

# Evaluamos el modelo en los datos aumentados
scores_aug = model_final.evaluate(X_augmented, y, verbose=0)
print("\n=== Evaluación en Datos Aumentados ===")
print(f"Loss: {scores_aug[0]:.4f} | Accuracy: {scores_aug[1]*100:.2f}%")

# Opcional: Generar reporte de clasificación y matriz de confusión con datos aumentados
y_pred = np.argmax(model_final.predict(X_augmented), axis=1)
print("\nReporte de clasificación (Datos Aumentados):")
print(classification_report(y, y_pred))
print("Matriz de confusión (Datos Aumentados):")
print(confusion_matrix(y, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
