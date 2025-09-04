# Preparación del Dataset de Entrenamiento para Enfermedades del Banano - 10 Clases

Para el dataset de entrenamiento, utilizamos un enfoque de **dataset dual** combinando el dataset BananaLSD original con fuentes adicionales, resultando en un dataset integral de **4,469 imágenes** distribuidas en **10 categorías de enfermedades**. Esta aproximación representa la colección más completa utilizada en estudios de aprendizaje profundo para patologías del banano hasta la fecha.

## Fuentes del Dataset

### Dataset Primario
- **BananaLSD (Banana Leaf Spot Diseases)** de Kaggle
- **Origen:** Bangladesh (condiciones de campo reales)
- **Características:** Imágenes capturadas con dispositivos móviles bajo iluminación natural
- **Total:** 1,600 imágenes base

### Dataset Secundario
- **Fuentes adicionales:** Repositorios públicos y colaboraciones institucionales
- **Diversidad geográfica:** Múltiples regiones productoras
- **Variabilidad de cultivares:** Diferentes variedades de banano
- **Total adicional:** 2,869 imágenes

## Estructura del Dataset Completo

```
/banana_disease_comprehensive_dataset/
├── train/ (3,576 imágenes - 80%)
│   ├── healthy/ (802 imágenes)
│   ├── insect_pest/ (482 imágenes)
│   ├── black_sigatoka/ (375 imágenes)
│   ├── cordana/ (320 imágenes)
│   ├── pestalotiopsis/ (320 imágenes)
│   ├── sigatoka/ (320 imágenes)
│   ├── moko_disease/ (308 imágenes)
│   ├── bract_virus/ (280 imágenes)
│   ├── panama_disease/ (230 imágenes)
│   └── yellow_sigatoka/ (139 imágenes)
└── validation/ (893 imágenes - 20%)
    ├── healthy/ (200 imágenes)
    ├── insect_pest/ (120 imágenes)
    ├── black_sigatoka/ (94 imágenes)
    ├── cordana/ (80 imágenes)
    ├── pestalotiopsis/ (80 imágenes)
    ├── sigatoka/ (80 imágenes)
    ├── moko_disease/ (77 imágenes)
    ├── bract_virus/ (70 imágenes)
    ├── panama_disease/ (57 imágenes)
    └── yellow_sigatoka/ (35 imágenes)
```

## Distribución Detallada por Clase

| Categoría de Enfermedad | Total | Train (80%) | Val (20%) | Porcentaje |
|-------------------------|-------|-------------|-----------|------------|
| Healthy (Sano) | 1,002 | 802 | 200 | 22.4% |
| Insect Pest (Plaga de Insectos) | 602 | 482 | 120 | 13.5% |
| Black Sigatoka (Sigatoka Negra) | 469 | 375 | 94 | 10.5% |
| Cordana | 400 | 320 | 80 | 9.0% |
| Pestalotiopsis | 400 | 320 | 80 | 9.0% |
| Sigatoka | 400 | 320 | 80 | 9.0% |
| Moko Disease (Enfermedad de Moko) | 385 | 308 | 77 | 8.6% |
| Bract Virus (Virus de Brácteas) | 350 | 280 | 70 | 7.8% |
| Panama Disease (Mal de Panamá) | 287 | 230 | 57 | 6.4% |
| Yellow Sigatoka (Sigatoka Amarilla) | 174 | 139 | 35 | 3.9% |
| **TOTAL** | **4,469** | **3,576** | **893** | **100%** |

## Características del Dataset

### Desbalance de Clases
- **Ratio máximo:** 5.76:1 (Healthy:Yellow Sigatoka)
- **Implicaciones:** Requiere estrategias específicas de mitigación durante entrenamiento
- **Reflejo realista:** La distribución representa prevalencias naturales en campo

### Especificaciones Técnicas
- **Resolución estándar:** 224×224 píxeles RGB
- **Formato:** JPG/PNG
- **Normalización:** Valores de píxel escalados a [0,1]
- **Método de redimensionamiento:** Interpolación bilineal con preservación de relación de aspecto

## Estrategias de Aumento de Datos Implementadas

### Transformaciones Geométricas
- Rotación aleatoria: ±20°
- Volteo horizontal: p=0.5
- Zoom aleatorio: 0.8-1.2×

### Aumentos de Color
- Ajuste de brillo: ±20%
- Modificación de contraste: ±15%
- Variación de saturación: ±25%

### Regularización Avanzada
- **Inyección de ruido gaussiano:** σ=0.01
- **Regularización Mixup:** α=0.2 para límites de decisión más suaves

## Criterios de Calidad de Imagen

Siguiendo los estándares establecidos por Barbedo (2018):
- **Nitidez:** Sin desenfoque significativo
- **Iluminación:** Rango dinámico adecuado
- **Integridad:** Sin corrupción de archivos
- **Relevancia:** Síntomas claramente visibles

## Consideraciones para Mejoras Futuras

### Limitaciones Identificadas
1. **Sesgo geográfico:** Predominancia de imágenes del sur de Asia
2. **Variabilidad de cultivares:** Representación limitada de variedades africanas
3. **Condiciones de captura:** Necesidad de mayor diversidad ambiental

### Expansiones Recomendadas
- Incorporación de imágenes de múltiples continentes
- Inclusión de diferentes variedades de Musa spp.
- Ampliación de condiciones de iluminación y clima
- Adición de enfermedades emergentes regionales

## División Estratificada

La división 80/20 mantiene las proporciones originales de cada clase, asegurando representatividad tanto en entrenamiento como en validación. Esta estrategia es crucial dado el desbalance significativo del dataset.

## Reproducibilidad

- **Semilla aleatoria fija:** 42 para todas las divisiones
- **Documentación completa:** Metadatos de cada imagen preservados
- **Trazabilidad:** Origen de cada muestra registrado

Este dataset representa el más comprehensivo utilizado para clasificación multi-enfermedad del banano en la literatura de aprendizaje profundo, superando significativamente trabajos previos en diversidad de patologías y tamaño de muestra.
