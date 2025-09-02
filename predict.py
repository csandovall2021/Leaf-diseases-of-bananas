import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Ruta al modelo entrenado
#MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'banana_disease_model_10classes.h5')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'banana_disease_model_transfer_learning.h5')

# Cargar el modelo entrenado
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Definir las clases (orden alfabético como las detecta Keras)
CLASS_NAMES = [
    "black_sigatoka", 
    "bract_virus", 
    "cordana",
    "healthy", 
    "insect_pest", 
    "moko_disease", 
    "panama_disease", 
    "pestalotiopsis",
    "sigatoka",
    "yellow_sigatoka"
]

# Mapeo para nombres más amigables
FRIENDLY_NAMES = {
    "black_sigatoka": "Sigatoka Negra",
    "yellow_sigatoka": "Sigatoka Amarilla", 
    "sigatoka": "Sigatoka",
    "bract_virus": "Virus de Brácteas",
    "insect_pest": "Plaga de Insectos",
    "healthy": "Hoja Sana",
    "moko_disease": "Enfermedad de Moko",
    "panama_disease": "Mal de Panamá",
    "cordana": "Cordana",
    "pestalotiopsis": "Pestalotiopsis"
}

def predict_image(img_path):
    if model is None:
        return {"error": "Modelo no cargado. Verifique la ruta del modelo."}

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)[0]
        
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        friendly_name = FRIENDLY_NAMES[predicted_class_name]
        confidence = float(predictions[predicted_class_index])

        # Interpretaciones detalladas
        interpretations = {
            "healthy": "La hoja de banano está sana. No se detectan signos de enfermedad.",
            "black_sigatoka": "Sigatoka Negra detectada. Enfermedad fúngica grave que requiere tratamiento fungicida inmediato.",
            "yellow_sigatoka": "Sigatoka Amarilla detectada. Menos agresiva que la negra, pero requiere monitoreo y posible tratamiento.",
            "sigatoka": "Sigatoka detectada. Enfermedad fúngica foliar que requiere tratamiento preventivo.",
            "bract_virus": "Virus de Brácteas detectado. Eliminar material infectado y controlar vectores.",
            "insect_pest": "Daño por plagas de insectos detectado. Implementar control integrado de plagas.",
            "moko_disease": "Enfermedad de Moko detectada. Bacteria muy contagiosa, cuarentena inmediata requerida.",
            "panama_disease": "Mal de Panamá detectado. Enfermedad vascular grave causada por Fusarium. Consultar especialista urgentemente.",
            "cordana": "Cordana detectada. Hongo que causa manchas foliares. Aplicar fungicidas preventivos.",
            "pestalotiopsis": "Pestalotiopsis detectada. Hongo oportunista que afecta hojas débiles. Mejorar condiciones de cultivo."
        }

        return {
            "result": friendly_name,
            "technical_name": predicted_class_name,
            "confidence": confidence,
            "interpretation": interpretations.get(predicted_class_name, "Consultar con especialista."),
            "raw_scores": {FRIENDLY_NAMES[cls]: float(score) for cls, score in zip(CLASS_NAMES, predictions)}
        }
    except Exception as e:
        return {"error": f"Error al procesar la imagen: {e}"}