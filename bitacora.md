# Bitácora de Desarrollo (Construir, Medir, Aprender)

### Metodología de Trabajo y *Vibe Engineering*

El desarrollo del proyecto siguió un ciclo iterativo de **Construir → Medir → Aprender**, utilizando asistentes de IA como apoyo para la modularización en `src/`, la implementación de funciones de auditoría y la depuración de errores de entorno. Cada etapa cerró con una verificación práctica antes de pasar a la siguiente.

---

### Etapa 1: Entorno de Desarrollo y Estructura Base del Proyecto

* **Objetivo:** Establecer un entorno reproducible en Python 3.12 con `uv` y aislar las dependencias y datos de prueba.
* **Construir:**
  * Creación de `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` y `AGENTS.md`.
  * Estructuración de directorios: `assets/ejemplos/` (capturas térmicas de referencia), `colab/ejecutar_en_colab.ipynb` (plan B con GPU) y el paquete modular `src/` (`__init__.py`, `diffusion.py`, `honesty.py`, `classifier.py`, `analysis.py`, `utils.py`).
* **Medir / Bloqueos Técnicos Superados:**
  1. En las pruebas iniciales sobre PowerShell en Windows surgieron errores con `uv` y rutas relativas (`FileNotFoundError`), lo que llevó temporalmente a usar un `.venv` manual y rutas absolutas con `pathlib.Path`.
  2. Durante la descarga de pesos desde Hugging Face ocurrió un error de espacio insuficiente en disco (`os error 112` al reconstruir checkpoints de ~4.27 GB).
  3. Al migrar el proyecto a una ruta limpia (`C:\Proyectos\auditoria-termografia-difusion`), se logró estandarizar el entorno reproducible con `uv` y Python 3.12.
* **Solución definitiva:** Estandarización mediante `uv sync` y `uv run`, verificando que `.gitignore` excluya `.venv/`, `__pycache__/`, `.env` y cachés de modelos.

---

### Etapa 2: Implementación del Pipeline Dual de Difusión

* **Objetivo:** Integrar dos modelos de difusión de Hugging Face (`diffusers`) para comparar su comportamiento sobre capturas infrarrojas de baja resolución (`160×120` px).
* **Construir:**
  * Módulo `src/diffusion.py` con dos pipelines `AutoPipelineForImage2Image`:
    1. `runwayml/stable-diffusion-v1-5` con scheduler `DPMSolverMultistepScheduler`.
    2. `stabilityai/sd-turbo` (destilación por consistencia en 1–4 pasos).
  * Caché en memoria mediante `_modelos_cargados` para no recargar pesos en cada interacción.
* **Medir / Bloqueo Técnico Superado:** Las capturas nativas de `160×120` px (y archivos `.bmp` con modos de color no RGB) provocaban fallos de formato y colapso en el decodificador VAE por las divisiones sucesivas por 8 en el espacio latente.
* **Solución:** Preprocesamiento estricto que convierte toda entrada a `"RGB"` y reescala a `512×512` px con interpolación Lanczos antes de ejecutar la difusión.

---

### Etapa 3: Auditoría Metrológica, Métricas y Detección de Discrepancias

* **Objetivo:** Medir y visualizar qué conserva el modelo y qué inventa respecto de la captura original del sensor.
* **Construir:**
  * `src/honesty.py`: Mapa de diferencias mediante resta matricial absoluta con realce para destacar píxeles alterados, más la advertencia metrológica obligatoria.
  * `src/classifier.py`: Auditoría preventiva con `MobileNetV2` sobre coherencia térmica.
  * `src/analysis.py`: Análisis visual cualitativo Zero-Shot con SigLIP (`google/siglip-base-patch16-224`).
  * `001_evaluar_metricas.py`: Script independiente de evaluación cuantitativa con **PSNR**, **SSIM** y **MAE** comparando ambos difusores contra interpolación bicúbica clásica.
* **Medir / Aprendizaje:**
  * **Faltante de dependencias:** Al integrar SigLIP en `002_app.py`, `SiglipTokenizer` lanzó `ImportError` por falta de `sentencepiece`; se agregaron `sentencepiece` y `protobuf` con `uv add` para mantener `pyproject.toml` y `uv.lock` reproducibles.
  * **Resultados cuantitativos reales en CPU (`001_evaluar_metricas.py`):**
    * *Bicúbica Clásica (Sin IA):* PSNR `51.18 dB` | SSIM `0.9993` | MAE `0.17`
    * *Modelo A (SD v1.5):* PSNR `22.36 dB` | SSIM `0.6966` | MAE `9.70`
    * *Modelo B (SD-Turbo):* PSNR `22.29 dB` | SSIM `0.7211` | MAE `10.12`
  * **Conclusión técnica:** `SD-Turbo` preservó mejor la estructura espacial (SSIM más alto), mientras que `SD v1.5` obtuvo un MAE levemente menor; sin embargo, ambos modelos se alejan drásticamente de la referencia bicúbica y alteran valores de píxeles. Además, `MobileNetV2` genérico arrojó confianzas bajas (~`33.9%`, *Dudosa / Revisión Manual*) tanto en originales como en salidas, confirmando que debe usarse solo como alerta preventiva junto con los mapas de diferencia.

---

### Etapa 4: Interfaz Unificada en Gradio, Pruebas de Detalles Inventados y Rebranding

* **Objetivo:** Integrar todos los módulos en una interfaz local ejecutable con `uv run 002_app.py`.
* **Construir:**
  * Desarrollo de `002_app.py` con `gr.Blocks`, carga de ejemplos de `assets/ejemplos/`, slider de intensidad (`Strength` acotado entre `0.10` y `0.50`), vista simultánea de **Original | SD v1.5 | SD-Turbo**, sus **Mapas de Diferencias** y el reporte técnico consolidado.
  * Soporte adaptativo de hardware: precisión reducida (`float16`) y *CPU offload* cuando hay GPU CUDA disponible (o mediante `colab/ejecutar_en_colab.ipynb`), y ejecución funcional en CPU (`float32`) para pruebas locales sin GPU dedicada.
  * Rebranding del proyecto hacia una **herramienta de evaluación comparativa y auditoría de discrepancias**.
* **Medir / Validación Final (casos reales documentados en `assets/`):**
  * A intensidad baja (`0.25`) sobre capturas con ruido, ambos modelos ya distorsionan la escala numérica lateral (`SSIM` ~`0.45` / `0.41`).
  * Al subir `Strength` a `0.50`, ambos modelos producen detalles inventados claros: `SD-Turbo` inventa patrones artificiales tipo panal sobre zonas ruidosas y un patrón circular falso en el pecho; `SD v1.5` inventa manchas rojas y rasgos en el rostro; y ambos deforman los números de la barra térmica.

---

### Estrategia de Priorización de Desarrollo

1. **Prioridad 1 — Funcionalidad Core:** Pipeline operativo de difusión (`src/diffusion.py`) y análisis visual con SigLIP (`src/analysis.py`), resolviendo el preprocesamiento RGB/Lanczos para el VAE.
2. **Prioridad 2 — Honestidad sobre lo Reconstruido:** Mapas de diferencias (`src/honesty.py`), control acotado de intensidad (`0.10–0.50`) y advertencia metrológica visible en pantalla.
3. **Prioridad 3 — Documentación y Demo:** Limpieza del repositorio, capturas recortadas en `assets/`, actualización de `README.md` (con la sección *"La paradoja entre lo real y lo inventado"*) y guion del video de 5 minutos.
4. **Prioridad 4 — Extensiones y Métricas:** Benchmark cuantitativo reproducible en `001_evaluar_metricas.py` (PSNR, SSIM, MAE vs. interpolación bicúbica) y auditoría preventiva con `MobileNetV2`.
