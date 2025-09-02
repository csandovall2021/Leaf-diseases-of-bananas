# Preparación del Dataset de Entrenamiento para Enfermedades del Banano

Para el dataset de entrenamiento, utilizaremos el `AugmentedSet` del dataset BananaLSD, ya que este conjunto ha sido enriquecido con técnicas de aumento de datos, lo que proporciona una mayor diversidad de imágenes y ayuda a mejorar la robustez del modelo de clasificación. Este conjunto contiene un total de 1600 imágenes, con 400 imágenes por clase (sana, cordana, pestalotiopsis y sigatoka).

## Estructura del Dataset

El dataset descomprimido tiene la siguiente estructura:

```
/home/ubuntu/banana_disease_dataset/
├── BananaLSD/
│   ├── AugmentedSet/
│   │   ├── healthy/
│   │   ├── cordana/
│   │   ├── pestalotiopsis/
│   │   └── sigatoka/
│   └── OriginalSet/
│       ├── healthy/
│       ├── cordana/
│       ├── pestalotiopsis/
│       └── sigatoka/
```

Para el entrenamiento, nos enfocaremos en el directorio `AugmentedSet`.

## Pasos para la Preparación del Dataset

1.  **Verificación de la integridad:** Aunque ya se ha descomprimido, se realizará una verificación rápida para asegurar que todas las imágenes estén presentes y no estén corruptas.
2.  **Organización:** El dataset ya está organizado en subcarpetas por clase, lo cual es ideal para el entrenamiento de modelos de aprendizaje profundo.
3.  **Preprocesamiento (si es necesario):** Las imágenes ya tienen una resolución estándar de 224x224 píxeles, lo que es un buen punto de partida. Durante el desarrollo del modelo, se pueden aplicar transformaciones adicionales en tiempo real (normalización, etc.) según los requisitos del modelo.

## Resumen del Dataset de Entrenamiento

*   **Origen:** Banana Leaf Spot Diseases (BananaLSD) Dataset (Kaggle)
*   **Conjunto utilizado:** `AugmentedSet`
*   **Número total de imágenes:** 1600
*   **Clases:**
    *   `healthy` (sana): 400 imágenes
    *   `cordana`: 400 imágenes
    *   `pestalotiopsis`: 400 imágenes
    *   `sigatoka`: 400 imágenes
*   **Resolución de imágenes:** 224x224 píxeles
*   **Formato:** JPG

Este dataset será la base para el desarrollo del modelo CNN en la siguiente fase. Es importante destacar que este dataset se enfoca en enfermedades foliares. Para incluir enfermedades causadas por nematodos o marchitez por Fusarium, se necesitaría buscar y fusionar datasets adicionales que contengan imágenes de esos síntomas, especialmente aquellos que se manifiestan en raíces o tallos, o síntomas foliares indirectos específicos de esas enfermedades. Por ahora, nos centraremos en las enfermedades cubiertas por este dataset.

