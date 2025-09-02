import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuración
base_dir = 'C:/wamp/www/FitoApp/BananaLSD/AugmentedSet'
MODEL_PATH = 'C:/wamp/www/FitoApp/modelos/banana_disease_model_10classes.h5'

# Cargar el modelo entrenado
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente")

# Crear generador para evaluación (igual que en entrenamiento)
eval_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generador de validación (mismo split que en entrenamiento)
validation_generator = eval_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Importante: sin shuffle para mantener orden
)

print(f"Clases encontradas: {list(validation_generator.class_indices.keys())}")
print(f"Muestras de validación: {validation_generator.samples}")

# Evaluar el modelo
print("\nEvaluando modelo...")
eval_result = model.evaluate(validation_generator, verbose=1)
print(f"Pérdida de validación: {eval_result[0]:.4f}")
print(f"Precisión de validación: {eval_result[1]:.4f}")
if len(eval_result) > 2:
    print(f"Top-3 accuracy: {eval_result[2]:.4f}")

# Generar predicciones
print("\nGenerando predicciones...")
validation_generator.reset()
Y_pred = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

# Obtener etiquetas verdaderas
y_true = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

# Generar reporte de clasificación
print('\n' + '='*60)
print('REPORTE DE CLASIFICACIÓN')
print('='*60)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Guardar reporte en archivo
with open('classification_report.txt', 'w', encoding='utf-8') as f:
    f.write('REPORTE DE CLASIFICACIÓN - ENFERMEDADES DEL BANANO\n')
    f.write('='*60 + '\n')
    f.write(f'Precisión de validación: {eval_result[1]:.4f}\n')
    if len(eval_result) > 2:
        f.write(f'Top-3 accuracy: {eval_result[2]:.4f}\n')
    f.write('\n' + report)

# Crear matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Graficar matriz de confusión
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names,
           yticklabels=class_names,
           cbar_kws={'label': 'Número de Muestras'})

plt.title('Matriz de Confusión - Clasificación de Enfermedades del Banano\n' + 
          f'Precisión Global: {eval_result[1]:.2%}', fontsize=16, pad=20)
plt.ylabel('Clase Real', fontsize=14)
plt.xlabel('Clase Predicha', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar matriz de confusión
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
print(f"\nMatriz de confusión guardada como 'confusion_matrix.png' y 'confusion_matrix.pdf'")

# Mostrar gráfico
plt.show()

# Calcular y mostrar métricas por clase
print('\n' + '='*60)
print('MÉTRICAS DETALLADAS POR CLASE')
print('='*60)

# Calcular precisión, recall y F1 por clase manualmente
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

metrics_df_data = []
for i, class_name in enumerate(class_names):
    metrics_df_data.append({
        'Clase': class_name,
        'Precisión': f"{precision[i]:.4f}",
        'Recall': f"{recall[i]:.4f}",
        'F1-Score': f"{f1[i]:.4f}",
        'Muestras': support[i]
    })

# Mostrar tabla de métricas
print(f"{'Clase':<18} {'Precisión':<10} {'Recall':<10} {'F1-Score':<10} {'Muestras':<10}")
print('-' * 70)
for data in metrics_df_data:
    print(f"{data['Clase']:<18} {data['Precisión']:<10} {data['Recall']:<10} {data['F1-Score']:<10} {data['Muestras']:<10}")

# Métricas globales
macro_avg_precision = np.mean(precision)
macro_avg_recall = np.mean(recall)
macro_avg_f1 = np.mean(f1)

print('\n' + '-' * 70)
print(f"{'Macro Average':<18} {macro_avg_precision:.4f}     {macro_avg_recall:.4f}     {macro_avg_f1:.4f}     {np.sum(support)}")

print(f"\n¡Evaluación completada!")
print(f"Archivos generados:")
print(f"- confusion_matrix.png")
print(f"- confusion_matrix.pdf") 
print(f"- classification_report.txt")