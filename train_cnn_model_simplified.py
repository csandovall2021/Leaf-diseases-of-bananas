import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

# Configuración
base_dir = 'C:/wamp/www/FitoApp/BananaLSD/AugmentedSet'
img_width, img_height = 224, 224
batch_size = 16  # Reducido para estabilidad
epochs = 15

# Generadores más conservadores
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print(f"Clases encontradas: {list(train_generator.class_indices.keys())}")
print(f"Muestras de entrenamiento: {train_generator.samples}")
print(f"Muestras de validación: {validation_generator.samples}")

# OPCIÓN 2: Transfer Learning (recomendado)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax') # Usar len(class_indices) para el número de clases
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]) # Especificar k=3 y un nombre para la métrica

# Callbacks más agresivos
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=0.00001, monitor='val_accuracy')
]

print(model.summary())

# Entrenamiento
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluar y guardar
model_save_path = 'C:/wamp/www/FitoApp/modelos/banana_disease_model_transfer_learning.h5' # Nuevo nombre para diferenciar
model.save(model_save_path)

# Métricas finales
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f'Precisión final de entrenamiento: {final_train_acc:.4f}')
print(f'Precisión final de validación: {final_val_acc:.4f}')

# Evaluar en conjunto de validación
# Asegúrate de que el orden de las variables coincida con el orden de las métricas en model.compile
val_loss, val_accuracy, val_top3_accuracy = model.evaluate(validation_generator)
print(f'Precisión en validación: {val_accuracy:.4f}')
print(f'Top-3 precisión en validación: {val_top3_accuracy:.4f}') # Usar el nombre correcto de la variable
