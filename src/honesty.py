import numpy as np
from PIL import Image

def calcular_mapa_diferencias(original: Image.Image, restaurada: Image.Image) -> Image.Image:
    """
    Resta la imagen procesada de la original para resaltar visualmente
    dónde el modelo de difusión alteró o inventó detalles térmicos.
    """
    if original is None or restaurada is None:
        return None

    # Redimensionar la original al tamaño de la restaurada para poder restar
    original_alineada = original.convert("RGB").resize(restaurada.size, Image.Resampling.BILINEAR)
    arr_orig = np.array(original_alineada, dtype=np.float32)
    arr_rest = np.array(restaurada.convert("RGB"), dtype=np.float32)

    diff = np.abs(arr_rest - arr_orig)
    diff_resaltada = np.clip(diff * 3.0, 0, 255).astype(np.uint8)

    return Image.fromarray(diff_resaltada)

def generar_advertencia_metrologica(intensidad: float) -> str:
    """
    Genera el aviso técnico y ético para el reporte de mantenimiento.
    """
    return (
        "⚠️ **Aviso de Integridad Térmica (Human-AI Interaction):**\n"
        f"La imagen restaurada fue procesada con un factor de difusión de `{intensidad:.2f}`. "
        "Las microtexturas y gradientes añadidos son reconstrucciones probabilísticas del modelo generativo "
        "y carecen de calibración física del sensor Testo 875-1i. "
        "Consulte el mapa de diferencias antes de diagnosticar puntos calientes (hotspots)."
    )