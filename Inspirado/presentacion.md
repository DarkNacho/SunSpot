---
marp: true
theme: default
author: Ignacio Martínez Hernández
paginate: true
math: mathjax
style: |
  .header {
    position: absolute;
    top: 10px;
    right: 10px;
    height: 60px;
  }
  section {
    background-image: url('assets/utalca.png');
    background-position: top right;
    background-repeat: no-repeat;
    background-size: 400px;
    padding-top: 30px;
  }
---

# La Importancia de las Manchas Solares

- Las manchas solares son fenómenos clave de la actividad solar.
- Su estudio es fundamental para entender y predecir el clima espacial.
- La actividad solar puede afectar la tecnología terrestre, incluyendo comunicaciones por satélite y redes eléctricas.

---

# El Archivo Histórico de OGAUC

- El Observatorio Geofísico y Astronómico de la Universidad de Coimbra (OGAUC) posee uno de los archivos de imágenes solares más antiguos y extensos.
- Este archivo data de 1926, ofreciendo una perspectiva única sobre la actividad solar a largo plazo.
- La gran cantidad de imágenes hace que la detección manual sea ineficiente.
- La detección automática es vital para analizar este vasto archivo y apoyar estudios longitudinales.

---

# Métodos Tradicionales de Detección

- Los enfoques iniciales se basaron en morfología matemática y el análisis de intensidad de píxeles.
- Lograron precisiones aceptables en ciertos casos.
- Sin embargo, estos métodos a menudo enfrentaron desafíos significativos.
- Las condiciones atmosféricas y las anotaciones en las imágenes complicaban su efectividad.

> Hay más métodos mencionados en el paper, sirve como estado del arte.

---

# El Conjunto de Datos del OGAUC

- Se utilizaron 2000 imágenes de continuo H-alfa del espectroheliógrafo OGAUC.
- Las imágenes fueron capturadas entre 2012 y 2019, cubriendo parte del ciclo solar 24.
- Las dimensiones de las imágenes son $1200 \times 1000$ píxeles.
- Las imágenes fueron obtenidas de un archivo público.

---

# Desafíos del Conjunto de Datos

- Las condiciones atmosféricas durante la captura afectaron la calidad de las imágenes.
- La composición de las imágenes en "slices" y la presencia de anotaciones en los bordes añaden complejidad.
- Estos factores hacen que la detección automática de manchas solares sea un reto significativo.

---

# Preparación del Conjunto de Datos

- Para abordar los desafíos, se crearon seis subconjuntos de datos (A-F).
- Se aplicaron técnicas de aumento de datos como volteo, rotación y cizallamiento para ampliar el conjunto.
- Se realizaron experimentos con el recorte de imágenes en algunos conjuntos (D, E, F).
- El recorte ayudó a eliminar áreas no relevantes y a mejorar el enfoque en las manchas solares.

---

# Configuración Experimental: YOLOv5

- Se eligió YOLOv5 debido a su excelente rendimiento en tareas de detección de objetos.
- Se probaron diferentes versiones del modelo para evaluar su eficacia:
  - YOLOv5s (pequeño)
  - YOLOv5m (medio)
  - YOLOv5l (grande)
- La organización y etiquetado de los datos se realizó utilizando una plataforma de anotación de datos.
- El entrenamiento de los modelos se llevó a cabo en un entorno de computación en la nube con recursos de GPU.

---

# Métricas de Evaluación

- Para evaluar el rendimiento de los modelos, se utilizaron métricas estándar de detección de objetos:
  - **Precisión (P):** Mide la exactitud de las detecciones positivas.
  - **Recuperación (R):** Mide la capacidad del modelo para encontrar todas las manchas solares.
  - **Precisión Media Promedio (mAP@.5):** La métrica principal, que ofrece una evaluación general del rendimiento del modelo con un umbral de confianza de 0.5.

---

# Resultados

> Insertar tablas de resultados

---

# Trabajo Futuro

- Continuar refinando los modelos YOLO para una mayor precisión y eficiencia.
- Explorar otras arquitecturas de aprendizaje profundo para comparar su rendimiento.
- Integrar el sistema de detección automática en un marco operativo de monitoreo solar.
- Aplicar esta metodología a conjuntos de datos de imágenes solares aún más grandes y diversos para validar su robustez.
