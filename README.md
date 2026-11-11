# 🔬 Banco de Ensayos: Comparativa de Difusión en Termografía Industrial

**Evaluación de Fidelidad Metrológica: Stable Diffusion v1.5 vs. SD-Turbo + Auditoría MobileNetV2 y SigLIP** [Rama • Planificación de Aplicación con IA.pdf, estado1.jpg]

---

## 📌 1.\. Declaración del Proyecto

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


## 🏗️ 2. Arquitectura del Sistema y Stack Tecnológico

El proyecto está diseñado bajo una arquitectura modular y desacoplada en Python para garantizar reproducibilidad, mantenibilidad e independencia funcional [Imagenes de trabajo_integrador_2.ipynb - Colab.pdf, Rama • Planificación de Aplicación con IA.pdf].

---

### 📂 Estructura del Repositorio

```text
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

### 🛠️ Componentes y Modelos Seleccionados

1. **Interfaz Web Local (Gradio):** Se implementó mediante `gr.Blocks` para orquestar la ejecución local con un solo comando (`python app.py`) . Se priorizó **Gradio frente a Streamlit** por su arquitectura basada en eventos reactivos (`.click()`), permitiendo mantener en memoria RAM/VRAM los pipelines pesados de difusión y análisis de forma persistente, sin reejecutar todo el script ante cada interacción del usuario. Su disposición en columnas permite al técnico comparar en una única pantalla el **Dato Original (160×120)** frente a las **Restauraciones A y B** y sus respectivos **Mapas de Diferencias** .
2. **Pipeline de Difusión A — Estándar Multipasos:** `runwayml/stable-diffusion-v1-5` utilizando el planificador `DPMSolverMultistepScheduler`.
3. **Pipeline de Difusión B — Destilado Rápido:** `stabilityai/sd-turbo` mediante destilación por consistencia (*Adversarial Diffusion Distillation*). Ejecuta inferencias en 1 a 4 pasos, evaluando la preservación de la macroestructura térmica original y reduciendo la invención arbitraria de patrones.
4. **Auditoría de Paleta Térmica:** Red convolucional `MobileNetV2` para corroborar de forma automática si la distribución cromática de la imagen resultante se mantiene dentro de los límites de una paleta radiométrica válida o si incurre en riesgo de descalibración.
5. **Análisis Visual Cualitativo:** Modelo multimodal `google/siglip-base-patch16-256-multilingual` ejecutado en modo *Zero-Shot* para clasificar el nivel de nitidez semántica y alertar sobre la aparición de artefactos generativos.

---

### 🔄 Flujo del Pipeline de Datos (*Data Pipeline Flow*)

```
Captura Térmica Infrarroja (160×120 px)
       │
       ▼
[ Preprocesamiento en src/diffusion.py ]
       ├── Conversión estricta a 3 canales RGB
       └── Reescalado con Lanczos a 512×512 (previene colapso del VAE)
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[ Inferencia Modelo A ]                  [ Inferencia Modelo B ]
Stable Diffusion v1.5                    SD-Turbo (1-4 pasos)
Strength: 0.10 - 0.50                    Strength: 0.10 - 0.50
       │                                        │
       ▼                                        ▼
Imagen Restaurada A                      Imagen Restaurada B
       │                                        │
       ├───────────────────┬────────────────────┤
       ▼                   ▼                    ▼
[ src/honesty.py ]   [ src/classifier.py ] [ src/analysis.py ]
Resta matricial:     Auditoría MobileNetV2: Clasificación SigLIP:
|Orig - Restaurada|  Veredicto de paleta    Certeza cualitativa
       │                   │                    │
       ▼                   ▼                    ▼
Mapas de Diferencias  Semáforo Metrológico  Reporte Semántico
(Píxeles alterados)   (Apto / Riesgo)       (Nitidez / Artefactos)
       │                   │                    │
       └───────────────────┴────────────────────┘
                           │
                           ▼
          Despliegue unificado en interfaz Gradio
```

---
        
## 👁️ Comparación Visual, Mapas de Diferencias y Honestidad en la Salida

Para satisfacer los lineamientos de *Human-AI Interaction* (HAI) y asegurar que el técnico no sea inducido a diagnósticos erróneos, la interfaz incorpora tres mecanismos obligatorios \:

### 1. Comparación Directa Lado a Lado (Original vs. Resultados)
La pantalla expone simultáneamente tres columnas: la captura nativa del detector ($160 \times 120\text{ px}$ ampliada) frente a las salidas de SD v1.5 y SD-Turbo ($512 \times 512\text{ px}$) \. Esto impide que el operador juzgue una imagen generativa sin contrastarla con el dato de origen \.

### 2. Mapas de Diferencias Matriciales
Implementados en `src/honesty.py`, calculan la sustracción absoluta píxel por píxel entre la entrada y cada reconstrucción:

$$\Delta I(x, y) = \vert I_{\text{original}}(x, y) - I_{\text{restaurada}}(x, y) \vert$$

Las discrepancias se presentan en un mapa donde el fondo negro certifica fidelidad metrológica y las tonalidades brillantes denuncian de forma inequívoca qué zonas fueron alteradas o inventadas por el modelo \.

### 3. Aviso Explícito de "Generado por IA"
Al pie del reporte técnico se emite una advertencia formal recordando al usuario que las isotermas y gradientes generados constituyen estimaciones probabilísticas de la red y no mediciones de temperatura calibradas por microbolómetros \.

---

## 🌟 Extensiones Avanzadas: Explicabilidad Multimodal (VLM) y Métricas Cuantitativas

Para superar los requerimientos base del MVP y dotar al banco de ensayos de rigor científico e industrial, se incorporaron dos módulos complementarios de evaluación \:

### 1. Explicabilidad Visual Asistida por VLM (*Vision-Language Model*)

* **Implementación:** Módulo `src/analysis.py` utilizando `google/siglip-base-patch16-256-multilingual` en modo *Zero-Shot* \.
* **Propósito en el flujo:** Traduce las características visuales a un diagnóstico comprensible para el operador técnico en planta:
  * Evalúa la certeza semántica de si la reconstrucción preserva la apariencia de una "captura termográfica industrial válida" o si presenta "artefactos generativos y ruido" \.
  * Articula su resultado con la red convolucional `MobileNetV2` (`src/classifier.py`) para emitir un semáforo de riesgo metrológico: **✅ Apto para informe visual** o **⚠️ Riesgo de descalibración térmica** \.

### 2. Evaluación Cuantitativa Independiente de Fidelidad (Métricas PDI)

Se diseñó un script de ensayo metrológico automatizado (`prueba_metricas.py`) basado en `scikit-image` que compara la interpolación matemática bicúbica tradicional frente a Stable Diffusion v1.5 y SD-Turbo \:

* **SSIM (*Structural Similarity Index Measure*):** Mide la preservación estructural y de contornos de las isotermas térmicas originales. Valores cercanos a $1.0$ indican mínima distorsión de las geometrías de calor \.
* **PSNR (*Peak Signal-to-Noise Ratio*):** Cuantifica la relación señal-ruido en escala logarítmica ($\text{dB}$) resultante del proceso generativo respecto a la referencia \.
* **MAE (*Mean Absolute Error*):** Calcula la desviación promedio absoluta entre los niveles de píxeles:

     $$\text{MAE} = \frac{1}{M \cdot N} \sum_{x=1}^{M} \sum_{y=1}^{N} \vert I_{\text{orig}}(x,y) - I_{\text{rest}}(x,y) \vert$$

Un MAE elevado evidencia que el modelo introdujo variaciones térmicas arbitrarias incompatibles con una medición física \.

> **Hallazgo cuantitativo:**  
> Mientras que la interpolación bicúbica clásica preserva un SSIM alto pero mantiene el desenfoque, **SD-Turbo logra un balance óptimo entre ganancia de nitidez y bajo MAE**, superando a SD v1.5, el cual penaliza en fidelidad estructural debido a su tendencia a inventar microtexturas inexistentes \.

---

## ⚡ Optimización y Mitigación de Restricciones Técnicas

* **Adaptación al Espacio Latente del VAE:** El detector de $160 \times 120\text{ px}$ generaba fallos de tensores nulos al pasar por las capas reductoras del decodificador VAE de Stable Diffusion \[error1.jpg, Rama • Planificación de Aplicación con IA.pdf\]. Se integró un remuestreo matemático previo a $512 \times 512\text{ px}$ con interpolación Lanczos para garantizar estabilidad numérica \.
* **Caché en Memoria (`_modelos_cargados`):** Los pesos se instancian una sola vez en un diccionario global dentro de `src/diffusion.py`, evitando la recarga innecesaria de modelos en cada consulta \.
* **Gestión de Memoria y GPU:** Empleo de precisión reducida `torch.float16` y descarga controlada hacia CPU mediante `enable_model_cpu_offload()` para garantizar ejecuciones sin desbordamiento de memoria VRAM \.
---

## ⚖️ 3. Qué es Real y Qué es Inventado (Análisis Crítico)

El núcleo conceptual del proyecto establece que **restaurar no es recuperar información perdida**.

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