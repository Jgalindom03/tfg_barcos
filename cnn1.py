import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import keras_tuner as kt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------
# Simulación de datos de ejemplo
# ---------------------------
# Supongamos que tenemos 4 matrices (una por cada grado de severidad) y cada una contiene:
n_samples_per_class = 100  # número de muestras por clase (ajusta según tus datos)
n_instancias = 300         # número de instancias (ejemplo; en tu caso puede ser 30000 u otro)
n_canales = 6              # número de canales

X = []
y = []
for label in range(4):
    # Se simulan datos aleatorios; a cada clase se le añade un offset para simular diferencias.
    X_class = np.random.rand(n_samples_per_class, n_instancias, n_canales) + label * 0.1
    X.append(X_class)
    y += [label] * n_samples_per_class

# Concatenar los datos de todas las clases
X = np.concatenate(X, axis=0)
y = np.array(y)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Definición del modelo CNN con hiperparámetros ajustables
# ---------------------------
def build_model(hp):
    model = keras.Sequential()
    
    # Primera capa convolucional 1D
    model.add(layers.Conv1D(filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
                            kernel_size=hp.Choice('kernel_size_1', values=[3, 5, 7]),
                            activation='relu',
                            input_shape=(n_instancias, n_canales)))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Opción de agregar una segunda capa convolucional
    if hp.Boolean('second_conv'):
        model.add(layers.Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
                                kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
                                activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
                           activation='relu'))
    
    # Capa de salida: 4 neuronas (una por cada clase) y activación softmax
    model.add(layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(
                    hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# Búsqueda de hiperparámetros con Hyperband
# ---------------------------
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='tuner_dir',
                     project_name='cnn_classification')

# Se inicia la búsqueda de los mejores hiperparámetros
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# Se obtiene el mejor modelo encontrado
best_model = tuner.get_best_models(num_models=1)[0]

# ---------------------------
# Evaluación del modelo
# ---------------------------
# Evaluar en el conjunto de prueba
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Predicción de clases para el conjunto de prueba
y_pred = np.argmax(best_model.predict(X_test), axis=1)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print("Balanced Accuracy:", balanced_acc)
