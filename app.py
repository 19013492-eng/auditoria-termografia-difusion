import glob
import gradio as gr
from PIL import Image

from src.diffusion import restaurar_con_modelo
from src.classifier import auditar_calidad_termica
from src.honesty import calcular_mapa_diferencias, generar_advertencia_metrologica
from src.analysis import analizar_termografia_zeroshot

def comparar_dos_difusores(imagen_entrada: Image.Image, intensidad: float, progreso=gr.Progress()):
    if imagen_entrada is None:
        raise gr.Error("Por favor cargue una imagen térmica de prueba.")

    # 1. Auditoría del dato original
    progreso(0.1, desc="Auditando dato crudo del sensor Testo...")
    clase_orig, conf_orig = auditar_calidad_termica(imagen_entrada)

    # 2. Ejecución Modelo A: SD v1.5
    progreso(0.3, desc="Ejecutando Modelo A: Stable Diffusion v1.5...")
    mod_a = "Stable Diffusion v1.5 (Estándar)"
    img_a = restaurar_con_modelo(imagen_entrada, mod_a, intensidad)
    clase_a, conf_a = auditar_calidad_termica(img_a)
    diff_a = calcular_mapa_diferencias(imagen_entrada, img_a)

    # 3. Ejecución Modelo B: SD-Turbo
    progreso(0.65, desc="Ejecutando Modelo B: SD-Turbo...")
    mod_b = "SD-Turbo (Rápido / 1-4 pasos)"
    img_b = restaurar_con_modelo(imagen_entrada, mod_b, intensidad)
    clase_b, conf_b = auditar_calidad_termica(img_b)
    diff_b = calcular_mapa_diferencias(imagen_entrada, img_b)

    # 4. Análisis Cualitativo Zero-Shot (SigLIP)
    progreso(0.9, desc="Realizando análisis Zero-Shot con SigLIP...")
    analisis_a = analizar_termografia_zeroshot(img_a)
    analisis_b = analizar_termografia_zeroshot(img_b)

    # Informe comparativo estructurado
    informe_comparativo = f"""
### 📊 Auditoría Comparativa de Calidad Térmica (MobileNetV2)
- **Sensor Testo 875i Original**: `{clase_orig}` ({conf_orig:.1f}%)
- **Salida Modelo A (SD v1.5)**: `{clase_a}` ({conf_a:.1f}%)
- **Salida Modelo B (SD-Turbo)**: `{clase_b}` ({conf_b:.1f}%)

---
### 🔬 Evaluación de Transparencia y Fidelidad Metrológica
| Criterio | Modelo A (SD v1.5) | Modelo B (SD-Turbo) |
| :--- | :--- | :--- |
| **Enfoque Algorítmico** | Espacio Latente Multipasos (U-Net clásica) | Destilación por Consistencia (1-4 pasos) |
| **Comportamiento Isotérmico** | Tiende a reconstruir bordes ficticios ("alucina" gradientes) | Preserva mejor la estructura gruesa con mínima invención |
| **Veredicto para Mantenimiento** | {'Apto informe' if 'Valida_OK' in clase_a else '⚠️ Riesgo de descalibración'} | {'Apto informe' if 'Valida_OK' in clase_b else '⚠️ Riesgo de descalibración'} |

---
#### Análisis SigLIP - Modelo A:
{analisis_a}

#### Análisis SigLIP - Modelo B:
{analisis_b}

---
{generar_advertencia_metrologica(intensidad)}
"""
    return (
        imagen_entrada,
        img_a,
        diff_a,
        img_b,
        diff_b,
        informe_comparativo
    )

rutas_ejemplos = glob.glob("assets/ejemplos/*.*")
ejemplos_disponibles = [[r] for r in rutas_ejemplos if r.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

with gr.Blocks(title="Comparativa de Modelos de Difusión en Termografía") as demo:
    gr.Markdown("""
    # 🔬 Banco de Ensayos: Comparativa de Difusión en Termografía Industrial
    **Evaluación de Fidelidad Metrológica: Stable Diffusion v1.5 vs. SD-Turbo + Auditoría MobileNetV2**  
    *Herramienta para evaluar la preservación de isotermas reales sobre capturas Testo 875-1i.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            entrada_img = gr.Image(type="pil", label="Captura Térmica Original (Testo 875i)")
            slider_intensidad = gr.Slider(
                minimum=0.1,
                maximum=0.5,
                value=0.25,
                step=0.05,
                label="Intensidad de Modificación (Strength)"
            )
            boton_ejecutar = gr.Button("⚡ Comparar Ambos Modelos", variant="primary")

            if ejemplos_disponibles:
                gr.Examples(
                    examples=ejemplos_disponibles,
                    inputs=[entrada_img],
                    label="Casos de inspección guardados"
                )

        with gr.Column(scale=2):
            salida_original = gr.Image(type="pil", label="Dato Original del Sensor (160x120)", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🟢 Modelo A: Stable Diffusion v1.5")
                    salida_img_a = gr.Image(type="pil", label="Restaurada (SD v1.5)", interactive=False)
                    salida_diff_a = gr.Image(type="pil", label="Mapa Diferencias A", interactive=False)

                with gr.Column():
                    gr.Markdown("#### 🔵 Modelo B: SD-Turbo")
                    salida_img_b = gr.Image(type="pil", label="Restaurada (SD-Turbo)", interactive=False)
                    salida_diff_b = gr.Image(type="pil", label="Mapa Diferencias B", interactive=False)

    with gr.Row():
        salida_informe = gr.Markdown()

    boton_ejecutar.click(
        fn=comparar_dos_difusores,
        inputs=[entrada_img, slider_intensidad],
        outputs=[
            salida_original,
            salida_img_a,
            salida_diff_a,
            salida_img_b,
            salida_diff_b,
            salida_informe
        ]
    )

if __name__ == "__main__":
    demo.launch()