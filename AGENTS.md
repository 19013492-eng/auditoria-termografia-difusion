## Descripción
Banco de ensayos local para evaluar y comparar modelos de difusión aplicados a capturas termográficas industriales (cámara Testo 875-1i, resolución nativa 160×120 px), verificando qué detalles conserva y cuáles inventa la IA.

## Stack Tecnológico
- **Lenguaje y gestor:** Python 3.12 con `uv` (`pyproject.toml` y `uv.lock`).
- **Interfaz:** Gradio (`gr.Blocks`).
- **Modelos de difusión (`diffusers`):** `runwayml/stable-diffusion-v1-5` y `stabilityai/sd-turbo`.
- **Análisis visual y auditoría:** `google/siglip-base-patch16-224` (`transformers`, requiere `sentencepiece` y `protobuf`) y clasificador MobileNetV2.
- **Métricas y procesamiento:** `pillow`, `numpy`, `scikit-image` (PSNR, SSIM, MAE).

## Comandos Principales
- Instalar/sincronizar entorno: `uv sync`
- Ejecutar evaluación cuantitativa de métricas: `uv run 001_evaluar_metricas.py`
- Levantar interfaz web local en Gradio: `uv run 002_app.py`
- Alternativa con GPU: `colab/ejecutar_en_colab.ipynb`

## Estructura del Repositorio
- `002_app.py`: Interfaz principal en Gradio (comparación original vs. SD v1.5 vs. SD-Turbo, mapas de diferencias y reporte).
- `001_evaluar_metricas.py`: Script de benchmark cuantitativo (interpolación bicúbica vs. difusión).
- `src/diffusion.py`: Carga en caché (`_modelos_cargados`), conversión RGB/reescalado compatible con VAE y ejecución de modelos.
- `src/honesty.py`: Cálculo de mapas de diferencias (`|I_orig - I_rest| * 3.0`) y aviso obligatorio de integridad térmica.
- `src/classifier.py`: Auditoría de coherencia de paleta térmica (MobileNetV2).
- `src/analysis.py`: Análisis visual cualitativo Zero-Shot con SigLIP.
- `src/utils.py`: Funciones auxiliares compartidas.
- `assets/ejemplos/`: Capturas térmicas de prueba exportadas en `.bmp` y `.jpg`.

## Reglas para el Agente
1. **Separación modular:** Mantener la lógica de difusión, análisis, clasificación y honestidad dentro de `src/`; no concentrar lógica de modelos en `002_app.py`.
2. **Carga única de modelos:** Reutilizar las instancias cacheadas en memoria sin recargar pesos en cada clic de la interfaz.
3. **Preprocesamiento seguro:** Convertir siempre las imágenes de entrada a `"RGB"` (especialmente archivos `.bmp` o `.png`) y mantener dimensiones múltiplos de 8 (máximo 512 px) para el VAE.
4. **Honestidad metrológica obligatoria:** Conservar siempre la vista de la captura original, los mapas de diferencias, el límite de intensidad (`0.10`–`0.50`) y la advertencia de que los detalles generados no tienen calibración física.
5. **Seguridad y limpieza:** Nunca incluir tokens ni claves en el código, y mantener `.venv/`, `__pycache__/`, `.env` y cachés de modelos fuera de Git mediante `.gitignore`.
6. **Manejo de errores:** Mostrar mensajes claros para la persona usuaria (técnico de mantenimiento) sin romper la interfaz ante formatos o tamaños inesperados.
