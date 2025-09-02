import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from itertools import cycle

# Configuración
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

base_dir = 'C:/wamp/www/FitoApp/BananaLSD/AugmentedSet'
MODEL_PATH = 'C:/wamp/www/FitoApp/modelos/banana_disease_model_transfer_learning.h5'
RESULTS_DIR = 'model_analysis_results'

# Crear directorio para resultados
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("=== ANÁLISIS COMPLETO DEL MODELO DE CLASIFICACIÓN DE ENFERMEDADES DEL BANANO ===\n")

# Cargar modelo
print("1. Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ Modelo cargado exitosamente\n")

# Configurar generadores de datos
eval_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = eval_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

train_generator = eval_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

class_names = list(validation_generator.class_indices.keys())
n_classes = len(class_names)

print(f"2. Dataset información:")
print(f"   - Clases: {n_classes}")
print(f"   - Muestras entrenamiento: {train_generator.samples}")
print(f"   - Muestras validación: {validation_generator.samples}")
print(f"   - Clases: {class_names}\n")

# ========================================
# 1. DISTRIBUCIÓN DEL DATASET
# ========================================
print("3. Generando gráfico de distribución del dataset...")

# Contar imágenes por clase
class_counts_train = []
class_counts_val = []
all_class_counts = []

for class_name in class_names:
    class_dir = os.path.join(base_dir, class_name)
    total_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    train_count = int(total_count * 0.8)  # 80% para entrenamiento
    val_count = total_count - train_count  # 20% para validación
    
    all_class_counts.append(total_count)
    class_counts_train.append(train_count)
    class_counts_val.append(val_count)

# Crear gráfico de distribución
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de barras total
x_pos = np.arange(len(class_names))
bars1 = ax1.bar(x_pos, class_counts_train, label='Entrenamiento', alpha=0.8, color='#2E8B57')
bars2 = ax1.bar(x_pos, class_counts_val, bottom=class_counts_train, label='Validación', alpha=0.8, color='#FF6347')

ax1.set_xlabel('Enfermedades/Clases')
ax1.set_ylabel('Número de Imágenes')
ax1.set_title('Distribución del Dataset por Clase')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Añadir valores en las barras
for i, (train_count, val_count, total) in enumerate(zip(class_counts_train, class_counts_val, all_class_counts)):
    ax1.text(i, total + 10, str(total), ha='center', va='bottom', fontweight='bold')

# Gráfico de pastel
colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
wedges, texts, autotexts = ax2.pie(all_class_counts, labels=class_names, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax2.set_title('Distribución Porcentual del Dataset')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/dataset_distribution.png', bbox_inches='tight')
plt.savefig(f'{RESULTS_DIR}/dataset_distribution.pdf', bbox_inches='tight')
plt.close()
print("✓ Gráfico de distribución guardado\n")

# ========================================
# 2. EVALUACIÓN DEL MODELO
# ========================================
print("4. Evaluando modelo...")
validation_generator.reset()
eval_result = model.evaluate(validation_generator, verbose=0)
val_loss, val_accuracy, val_top3_accuracy = eval_result

print(f"   - Pérdida de validación: {val_loss:.4f}")
print(f"   - Precisión de validación: {val_accuracy:.4f}")
print(f"   - Top-3 accuracy: {val_top3_accuracy:.4f}\n")

# ========================================
# 3. PREDICCIONES Y MATRIZ DE CONFUSIÓN
# ========================================
print("5. Generando predicciones...")
validation_generator.reset()
Y_pred_proba = model.predict(validation_generator, verbose=0)
y_pred = np.argmax(Y_pred_proba, axis=1)
y_true = validation_generator.classes

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Gráfico de matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           square=True, linewidths=0.5, cbar_kws={'label': 'Número de Muestras'})

plt.title(f'Matriz de Confusión\nPrecisión Global: {val_accuracy:.2%}', fontsize=16, pad=20)
plt.ylabel('Clase Real', fontsize=12)
plt.xlabel('Clase Predicha', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', bbox_inches='tight')
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.pdf', bbox_inches='tight')
plt.close()
print("✓ Matriz de confusión guardada\n")

# ========================================
# 4. MÉTRICAS DETALLADAS POR CLASE
# ========================================
print("6. Calculando métricas por clase...")
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

# Crear DataFrame con métricas
metrics_data = []
for i, class_name in enumerate(class_names):
    metrics_data.append({
        'Clase': class_name,
        'Precisión': precision[i],
        'Recall': recall[i],
        'F1-Score': f1[i],
        'Soporte': support[i],
        'Muestras_Dataset': all_class_counts[i]
    })

metrics_df = pd.DataFrame(metrics_data)

# Guardar métricas en CSV
metrics_df.to_csv(f'{RESULTS_DIR}/metrics_by_class.csv', index=False)

# Crear gráfico de métricas por clase
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Precisión por clase
bars1 = ax1.bar(range(len(class_names)), precision, color='#4CAF50', alpha=0.8)
ax1.set_title('Precisión por Clase')
ax1.set_ylabel('Precisión')
ax1.set_xticks(range(len(class_names)))
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)
for i, v in enumerate(precision):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# Recall por clase
bars2 = ax2.bar(range(len(class_names)), recall, color='#2196F3', alpha=0.8)
ax2.set_title('Recall por Clase')
ax2.set_ylabel('Recall')
ax2.set_xticks(range(len(class_names)))
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
for i, v in enumerate(recall):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# F1-Score por clase
bars3 = ax3.bar(range(len(class_names)), f1, color='#FF9800', alpha=0.8)
ax3.set_title('F1-Score por Clase')
ax3.set_ylabel('F1-Score')
ax3.set_xticks(range(len(class_names)))
ax3.set_xticklabels(class_names, rotation=45, ha='right')
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)
for i, v in enumerate(f1):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# Soporte (número de muestras) por clase
bars4 = ax4.bar(range(len(class_names)), support, color='#9C27B0', alpha=0.8)
ax4.set_title('Número de Muestras de Validación por Clase')
ax4.set_ylabel('Número de Muestras')
ax4.set_xticks(range(len(class_names)))
ax4.set_xticklabels(class_names, rotation=45, ha='right')
ax4.grid(True, alpha=0.3)
for i, v in enumerate(support):
    ax4.text(i, v + 1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/metrics_by_class.png', bbox_inches='tight')
plt.savefig(f'{RESULTS_DIR}/metrics_by_class.pdf', bbox_inches='tight')
plt.close()
print("✓ Métricas por clase guardadas\n")

# ========================================
# 5. CURVAS PRECISION-RECALL
# ========================================
print("7. Generando curvas Precision-Recall...")

# Binarizar etiquetas para multiclase
y_true_bin = label_binarize(y_true, classes=range(n_classes))

plt.figure(figsize=(12, 8))
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal',
               'red', 'yellow', 'green', 'pink', 'brown'])

for i, (class_name, color) in enumerate(zip(class_names, colors)):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin[:, i], Y_pred_proba[:, i])
    plt.plot(recall_curve, precision_curve, color=color, lw=2, 
             label=f'{class_name} (AP={auc(recall_curve, precision_curve):.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curvas Precision-Recall por Clase')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/precision_recall_curves.png', bbox_inches='tight')
plt.savefig(f'{RESULTS_DIR}/precision_recall_curves.pdf', bbox_inches='tight')
plt.close()
print("✓ Curvas Precision-Recall guardadas\n")

# ========================================
# 6. CURVAS ROC
# ========================================
print("8. Generando curvas ROC...")

plt.figure(figsize=(12, 8))
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal',
               'red', 'yellow', 'green', 'pink', 'brown'])

for i, (class_name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], Y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{class_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/roc_curves.png', bbox_inches='tight')
plt.savefig(f'{RESULTS_DIR}/roc_curves.pdf', bbox_inches='tight')
plt.close()
print("✓ Curvas ROC guardadas\n")

# ========================================
# 7. REPORTE COMPLETO
# ========================================
print("9. Generando reporte completo...")

# Reporte de clasificación detallado
report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)

# Guardar reporte completo
with open(f'{RESULTS_DIR}/complete_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("ANÁLISIS COMPLETO DEL MODELO DE CLASIFICACIÓN DE ENFERMEDADES DEL BANANO\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. INFORMACIÓN DEL DATASET\n")
    f.write("-"*30 + "\n")
    f.write(f"Total de clases: {n_classes}\n")
    f.write(f"Total de imágenes: {sum(all_class_counts)}\n")
    f.write(f"Imágenes de entrenamiento: {train_generator.samples}\n")
    f.write(f"Imágenes de validación: {validation_generator.samples}\n\n")
    
    f.write("Distribución por clase:\n")
    for i, class_name in enumerate(class_names):
        f.write(f"  - {class_name}: {all_class_counts[i]} imágenes totales\n")
    
    f.write(f"\n2. MÉTRICAS GLOBALES DEL MODELO\n")
    f.write("-"*35 + "\n")
    f.write(f"Pérdida de validación: {val_loss:.4f}\n")
    f.write(f"Precisión de validación: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
    f.write(f"Top-3 Accuracy: {val_top3_accuracy:.4f} ({val_top3_accuracy*100:.2f}%)\n\n")
    
    f.write("3. MÉTRICAS POR CLASE\n")
    f.write("-"*22 + "\n")
    f.write(f"{'Clase':<18} {'Precisión':<10} {'Recall':<10} {'F1-Score':<10} {'Soporte':<10}\n")
    f.write("-" * 70 + "\n")
    
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name:<18} {precision[i]:<10.4f} {recall[i]:<10.4f} "
                f"{f1[i]:<10.4f} {support[i]:<10}\n")
    
    f.write("-" * 70 + "\n")
    f.write(f"{'Macro avg':<18} {np.mean(precision):<10.4f} {np.mean(recall):<10.4f} "
            f"{np.mean(f1):<10.4f} {np.sum(support):<10}\n")
    f.write(f"{'Weighted avg':<18} {report['weighted avg']['precision']:<10.4f} "
            f"{report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f} "
            f"{np.sum(support):<10}\n\n")
    
    f.write("4. ARCHIVOS GENERADOS\n")
    f.write("-"*20 + "\n")
    f.write("- dataset_distribution.png/pdf: Distribución del dataset\n")
    f.write("- confusion_matrix.png/pdf: Matriz de confusión\n")
    f.write("- metrics_by_class.png/pdf: Métricas por clase\n")
    f.write("- precision_recall_curves.png/pdf: Curvas Precision-Recall\n")
    f.write("- roc_curves.png/pdf: Curvas ROC\n")
    f.write("- metrics_by_class.csv: Datos de métricas en CSV\n")
    f.write("- complete_analysis_report.txt: Este reporte\n")

print("✓ Reporte completo guardado\n")

print("="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)
print(f"Todos los archivos guardados en: {RESULTS_DIR}/")
print(f"\nRESUMEN DE RESULTADOS:")
print(f"- Precisión de validación: {val_accuracy:.2%}")
print(f"- Top-3 Accuracy: {val_top3_accuracy:.2%}")
print(f"- Número de clases: {n_classes}")
print(f"- Total de imágenes: {sum(all_class_counts)}")
print(f"\nArchivos generados:")
print("- 6 gráficos (PNG y PDF)")
print("- 1 archivo CSV con métricas")
print("- 1 reporte completo de texto")
print("="*80)