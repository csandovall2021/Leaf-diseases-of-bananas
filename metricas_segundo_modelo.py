import matplotlib.pyplot as plt
import pickle
import os

# Ruta donde se guardó el historial
history_load_path = 'C:/wamp/www/FitoApp/modelos/training_history.pkl'

# Cargar el historial de entrenamiento
try:
    with open(history_load_path, 'rb') as file_pi:
        loaded_history = pickle.load(file_pi)
except FileNotFoundError:
    print(f"Error: El archivo de historial no se encontró en {history_load_path}")
    exit()

# Supongamos que `loaded_history` es el diccionario cargado
accuracy = loaded_history['accuracy']
val_accuracy = loaded_history['val_accuracy']
loss = loaded_history['loss']
val_loss = loaded_history['val_loss']
epochs = range(1, len(accuracy) + 1)

# Grafica la precisión
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, label='Precisión de entrenamiento')
plt.plot(epochs, val_accuracy, label='Precisión de validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Grafica la pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, label='Pérdida de validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Muestra las gráficas
plt.tight_layout()
plt.show()
