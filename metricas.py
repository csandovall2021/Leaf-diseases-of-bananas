import matplotlib.pyplot as plt

# Supongamos que `history` es el objeto devuelto por `model.fit()`
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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