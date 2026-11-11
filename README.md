# 🔬 Banco de Ensayos: Comparativa de Difusión en Termografía Industrial

**Evaluación de Fidelidad Metrológica: Stable Diffusion v1.5 vs. SD-Turbo + Auditoría MobileNetV2 y SigLIP** [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg]

---

## 📌 1\. Declaración del Proyecto

### Declaración del Proyecto

**"Este sistema permite que un técnico, durante la realización de sus tareas en una planta industrial, evalúe y mejore la legibilidad visual de imágenes termográficas de baja resolución nativa (160×120 px) para informes técnicos, sin incurrir en falsos diagnósticos ni caer en anomalías térmicas inventadas por modelos generativos de IA. La herramienta advierte activamente al operador las diferencias entre las imágenes evaluadas para detectar con precisión dónde el modelo realizó modificaciones frente a la medición directa del sensor infrarrojo."** 

### User Persona


* **Nombre y Perfil:** **Marcelo Gómez** (26 años), Técnico Electromecánico.
* **Herramienta de Trabajo:** Cámara termográfica **Testo 875-1i** (detector de plano focal FPA de 160 × 120 píxeles, sensibilidad térmica; 50 mK).
* **Contexto de Operación:** Realiza relevamientos periódicos en tableros de distribución eléctrica, centros de transformación y paneles solares para detectar puntos calientes (*hotspots*) antes de paradas de planta no programadas.
* **Problemática actual:**
  1. **Baja resolución del sensor:** La resolución nativa de 160×120 px luce pixelada al pegarla en reportes de auditoría ejecutivos.
  2. **Limitación de filtros tradicionales:** Los métodos clásicos de interpolación aumentan el tamaño pero difuminan y desenfocan los límites térmicos.
  3. **Riesgo crítico de la IA generativa:** Los modelos generativos convencionales tienden a alucinar texturas y gradientes continuos, lo cual puede disipar una anomalía real o generar un falso sobrecalentamiento en componentes críticos.
* **Necesidad Principal:** Una herramienta ágil y ejecutable en su jornada que optimice la nitidez de las capturas pero audite píxel a píxel las diferencias con el sensor real, delimitando los desvíos respecto a la medición física.


## 🏗️ 2\. Arquitectura del Sistema y Stack Tecnológico

El proyecto está diseñado bajo una arquitectura modular y desacoplada en Python para garantizar reproducibilidad, mantenibilidad e independencia funcional:
```
auditoria-termografia-difusion/
├── assets/
│   └── ejemplos/               # Muestras térmicas nativas del sensor FPA (160x120 px)
├── src/
│   ├── __init__.py
│   ├── diffusion.py            # Pipelines de difusión (SD v1.5 y SD-Turbo) con VAE a 512x512
│   ├── classifier.py           # Auditoría de coherencia de paleta térmica (MobileNetV2)
│   ├── analysis.py             # Clasificación cualitativa Zero-Shot (SigLIP)
│   └── honesty.py              # Cálculo de matrices de diferencia y advertencias metrológicas
├── app.py                      # Interfaz interactiva local desarrollada en Gradio
├── pyproject.toml              # Definición de entorno reproducible y dependencias
├── bitacora.md                 # Registro del proceso iterativo de desarrollo y decisiones
└── README.md                   # Documentación principal del proyecto
```
---

## ⚖️ 3. Qué es Real y Qué es Inventado (Análisis Crítico)

El núcleo conceptual del proyecto establece que **restaurar no es recuperar información perdida** [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf].

| Dimensión | Dato Real (Sensor FPA Testo) | Dato Inventado (Modelos de Difusión) |
| :--- | :--- | :--- |
| **Resolución Espacial** | Matriz física de $160 \times 120\text{ píxeles}$. Cada punto es una lectura de un microbolómetro [testo-875-ficha-tecnica.pdf, Rama • Planificación de Aplicación con IA.pdf]. | Matriz escalada a $512 \times 512\text{ píxeles}$. Los 243.200 píxeles adicionales son inferencias estadísticas de la red [Rama • Planificación de Aplicación con IA.pdf]. |
| **Información Radiométrica** | Medición física de temperatura calibrada (ej. rango $24.4\text{ °C} - 26.1\text{ °C}$) [testo-875-ficha-tecnica.pdf, estado1.jpg]. | Valores cromáticos en RGB generados por el decodificador VAE para coherencia visual [Rama • Planificación de Aplicación con IA.pdf]. |
| **Isotermas y Contornos** | Discontinuos, con grano y ruido característico del sensor térmico [testo-875-ficha-tecnica.pdf, Rama • Planificación de Aplicación con IA.pdf]. | Contornos suavizados y transiciones continuas sintetizadas artificialmente [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg]. |

### Comportamiento Comparativo de los Modelos:
- **Stable Diffusion v1.5:** Tiende a reconstruir bordes ficticios y alucinar microgradientes [Rama • Planificación de Aplicación con IA.pdf]. En el Mapa de Diferencias A, la dispersión de puntos brillantes evidencia una alteración notable del patrón original, introduciendo riesgo de descalibración metrológica [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].
- **SD-Turbo:** Al apoyarse en destilación por consistencia, mantiene anclada la distribución macro del calor [Rama • Planificación de Aplicación con IA.pdf]. Su Mapa de Diferencias B refleja intervenciones acotadas al filtrado de ruido, minimizando la invención de formas inexistentes [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].

> **Criterio Operativo en Planta:**  
> La imagen procesada por difusión es exclusivamente un **apoyo estético para reportes**. Cualquier decisión de intervención, diagnóstico de *hotspot* o cálculo de gradientes debe basarse en el **dato crudo original del sensor Testo** [Infrared computer vision for utility scale photovoltaic array inspection.pdf, Rama • Planificación de Aplicación con IA.pdf].

---

## 🤝 4. Principios de Human-AI Interaction (HAI)

- **Transparencia Activa:** No se oculta la imagen original; se presenta en pantalla junto con las restauraciones y los mapas de diferencias [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, estado1.jpg].
- **Control del Operador:** Se provee un deslizador (*Slider*) de intensidad (*Strength*) restringido a valores seguros ($0.10$ a $0.50$) para evitar alucinaciones descontroladas [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].
- **Explicabilidad del Diagnóstico:** Se genera un informe técnico automatizado que desglosa el veredicto de la auditoría y la certeza cualitativa [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].
- **Honestidad en la Salida:** Se emite una advertencia explícita recordando que la imagen resultante no constituye una medición certificada [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].

---

## 📐 5. Conceptos de Procesamiento Digital de Imágenes (PDI) Aplicados

- **Muestreo y Resolución Espacial:** Caracterización del submuestreo nativo del detector de $160 \times 120\text{ px}$ [testo-875-ficha-tecnica.pdf, Rama • Planificación de Aplicación con IA.pdf].
- **Interpolación Espacial:** Comparación de remuestreo matemático clásico (Lanczos/Bicúbica) contra super-resolución generativa [Rama • Planificación de Aplicación con IA.pdf].
- **Operaciones Matriciales Aritméticas:** Sustracción de imágenes normalizadas ($|I_{\text{orig}} - I_{\text{rest}}|$) con realce cromático en `src/honesty.py` para aislar discrepancias [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].
- **Reducción de Ruido (*Denoising*):** Filtrado de ruido térmico mediante difusión probabilística guiada por texto [Rama • Planificación de Aplicación con IA.pdf].

---

## 🚀 6. Instalación y Ejecución Local

### Prerrequisitos
- Python 3.12 instalado [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf].
- Git [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf].

### Pasos de Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd tpi2-termografia-restauracion
   ```

2. **Crear y activar el entorno virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate   # En Windows PowerShell
   ```

3. **Instalar dependencias:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # Si se cuenta con GPU NVIDIA
   pip install diffusers transformers accelerate gradio pillow scikit-image
   ```

4. **Ejecutar la aplicación:**
   ```bash
   python app.py
   ```

5. **Acceder a la interfaz web local:**
   Navegar a `http://localhost:7860` [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, estado1.jpg].

---

## 📺 7. Video Demostrativo (5 Minutos)

- **Enlace:** [Ver Video Demostrativo](#) [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf]

### Estructura del video:
- **00:00 - 01:00:** Presentación del problema en termografía ($160 \times 120\text{ px}$) y perfil de Martín Gómez [testo-875-ficha-tecnica.pdf, Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, Rama • Planificación de Aplicación con IA.pdf].
- **01:00 - 02:30:** Demostración en Gradio con inferencia comparativa (SD v1.5 vs. SD-Turbo) [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].
- **02:30 - 03:45:** Inspección de Mapas de Diferencias y auditoría MobileNetV2 / SigLIP [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg].
- **03:45 - 05:00:** Conclusiones metrológicas: por qué restaurar no es recuperar y cómo usar la herramienta de forma segura [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, Rama • Planificación de Aplicación con IA.pdf].

---

## 🤖 8. Declaración sobre el Uso de Asistentes de IA

Siguiendo las directrices de *Vibe / Agentic Engineering* [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf]:

- **Generación Asistida:** Asistencia en la modularización del código, implementación matemática del mapa de diferencias y resolución de compatibilidad con tensores en el VAE de Diffusers [Rama • Planificación de Aplicación con IA.pdf].
- **Decisiones Humanas del Autor:** Elección del caso de uso de termografía industrial con la cámara Testo 875-1i, diseño del banco de ensayos comparativo, formulación de los prompts de inspección y definición de las restricciones de uso seguro para mantenimiento predictivo [testo-875-ficha-tecnica.pdf, Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, Rama • Planificación de Aplicación con IA.pdf].

> Para consultar los detalles del proceso iterativo, revisar el archivo `bitacora.md` [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf].