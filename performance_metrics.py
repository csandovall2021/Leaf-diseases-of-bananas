# performance_metrics.py
import tensorflow as tf
import numpy as np
import time
import os
import random
import psutil
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'C:/wamp/www/FitoApp/modelos/banana_disease_model_transfer_learning.h5'
BASE_DIR = 'C:/wamp/www/FitoApp/BananaLSD/AugmentedSet'

def measure_model_size():
    """Medir tamaño del modelo en MB"""
    size_bytes = os.path.getsize(MODEL_PATH)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Tamaño del modelo: {size_mb:.2f} MB")
    return size_mb

def get_random_test_image():
    """Obtener una imagen aleatoria del dataset para pruebas"""
    # Elegir una carpeta aleatoria
    folders = ['black_sigatoka', 'bract_virus', 'cordana', 'healthy', 
               'insect_pest', 'moko_disease', 'panama_disease', 
               'pestalotiopsis', 'sigatoka', 'yellow_sigatoka']
    
    random_folder = random.choice(folders)
    folder_path = os.path.join(BASE_DIR, random_folder)
    
    # Obtener lista de imágenes en la carpeta
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if images:
        random_image = random.choice(images)
        image_path = os.path.join(folder_path, random_image)
        print(f"Imagen de prueba seleccionada: {random_folder}/{random_image}")
        return image_path
    return None

def measure_inference_time(model, test_image_path=None, n_runs=100):
    """Medir tiempo promedio de inferencia"""
    if test_image_path is None:
        test_image_path = get_random_test_image()
    
    if test_image_path is None or not os.path.exists(test_image_path):
        print("Error: No se pudo encontrar imagen de prueba")
        return None, None
    
    times = []
    
    # Primera ejecución para calentar
    img = image.load_img(test_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    _ = model.predict(img_array, verbose=0)
    
    # Mediciones reales
    for _ in range(n_runs):
        img = image.load_img(test_image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        start = time.time()
        predictions = model.predict(img_array, verbose=0)
        end = time.time()
        
        times.append((end - start) * 1000)  # convertir a ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"Tiempo de inferencia:")
    print(f"  Promedio: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Mínimo: {min_time:.2f} ms")
    print(f"  Máximo: {max_time:.2f} ms")
    
    return avg_time, std_time

def measure_throughput(model, batch_sizes=[1, 8, 16, 32]):
    """Medir throughput para diferentes tamaños de batch"""
    results = {}
    
    for batch_size in batch_sizes:
        dummy_input = np.random.rand(batch_size, 224, 224, 3)
        
        # Calentamiento
        for _ in range(5):
            model.predict(dummy_input, verbose=0)
        
        # Medición real
        start = time.time()
        n_iterations = 50
        for _ in range(n_iterations):
            model.predict(dummy_input, verbose=0)
        end = time.time()
        
        total_images = n_iterations * batch_size
        total_time = end - start
        throughput = total_images / total_time
        
        results[batch_size] = throughput
        print(f"Throughput (batch={batch_size}): {throughput:.2f} img/s")
    
    return results

def measure_memory_usage():
    """Medir uso de memoria del modelo"""
    process = psutil.Process(os.getpid())
    
    # Memoria inicial
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # Cargar modelo
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Memoria después
    mem_after = process.memory_info().rss / 1024 / 1024
    memory_used = mem_after - mem_before
    
    print(f"Memoria utilizada: {memory_used:.2f} MB")
    return memory_used, model

def test_on_multiple_images(model, n_images=10):
    """Probar el modelo en múltiples imágenes aleatorias"""
    print(f"\nProbando en {n_images} imágenes aleatorias...")
    times = []
    
    for i in range(n_images):
        image_path = get_random_test_image()
        if image_path:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            start = time.time()
            _ = model.predict(img_array, verbose=0)
            end = time.time()
            
            times.append((end - start) * 1000)
    
    if times:
        print(f"Tiempo promedio en {n_images} imágenes: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    
    return times

if __name__ == "__main__":
    print("="*60)
    print("ANÁLISIS DE RENDIMIENTO DEL MODELO")
    print("="*60)
    
    # Verificar que el directorio base existe
    if not os.path.exists(BASE_DIR):
        print(f"Error: No se encuentra el directorio {BASE_DIR}")
        exit(1)
    
    # 1. Tamaño del modelo
    print("\n1. TAMAÑO DEL MODELO")
    print("-"*30)
    model_size = measure_model_size()
    
    # 2. Cargar modelo
    print("\n2. CARGANDO MODELO")
    print("-"*30)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente")
    
    # 3. Tiempo de inferencia
    print("\n3. TIEMPO DE INFERENCIA")
    print("-"*30)
    avg_time, std_time = measure_inference_time(model)
    
    # 4. Throughput
    print("\n4. THROUGHPUT")
    print("-"*30)
    throughput_results = measure_throughput(model)
    
    # 5. Memoria
    print("\n5. USO DE MEMORIA")
    print("-"*30)
    memory_used, _ = measure_memory_usage()
    
    # 6. Prueba en múltiples imágenes
    print("\n6. PRUEBA EN MÚLTIPLES IMÁGENES")
    print("-"*30)
    multi_times = test_on_multiple_images(model, n_images=20)
    
    # Guardar resultados
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    with open('performance_results.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS DE RENDIMIENTO DEL MODELO\n")
        f.write("="*60 + "\n\n")
        f.write("CONFIGURACIÓN:\n")
        f.write(f"Modelo: {MODEL_PATH}\n")
        f.write(f"Dataset: {BASE_DIR}\n\n")
        
        f.write("MÉTRICAS DE RENDIMIENTO:\n")
        f.write("-"*30 + "\n")
        f.write(f"Tamaño del modelo: {model_size:.2f} MB\n")
        
        if avg_time is not None:
            f.write(f"Tiempo de inferencia (promedio): {avg_time:.2f} ± {std_time:.2f} ms\n")
        
        f.write(f"Memoria RAM utilizada: {memory_used:.2f} MB\n\n")
        
        f.write("THROUGHPUT (imágenes/segundo):\n")
        f.write("-"*30 + "\n")
        for batch, tput in throughput_results.items():
            f.write(f"  Batch size {batch}: {tput:.2f} img/s\n")
        
        f.write(f"\nRendimiento en CPU: Procesamiento individual a {throughput_results[1]:.2f} img/s\n")
        f.write(f"Rendimiento máximo: {max(throughput_results.values()):.2f} img/s con batch={max(throughput_results, key=throughput_results.get)}\n")
    
    # Imprimir resumen en consola
    print(f"\nTamaño del modelo: {model_size:.2f} MB")
    if avg_time is not None:
        print(f"Tiempo de inferencia: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Memoria utilizada: {memory_used:.2f} MB")
    print(f"Throughput máximo: {max(throughput_results.values()):.2f} img/s")
    
    print("\n✓ Resultados guardados en: performance_results.txt")