"""
Módulo: src/utils.py
Métricas cuantitativas de preservación estructural y calibración térmica.
Traduce diferencias matemáticas a impacto térmico operativo.
"""

from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def asegurar_dimensiones(imagen: Image.Image, tamaño: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Estandariza dimensiones usando interpolación bicúbica para comparar matrices."""
    if imagen.size != tamaño:
        return imagen.resize(tamaño, Image.Resampling.BICUBIC)
    return imagen


def calcular_metricas_fidelidad(
    img_referencia: Image.Image, 
    img_generada: Image.Image, 
    sensibilidad_c_por_nivel: float = 0.4
) -> Dict[str, any]:
    """
    Calcula SSIM, MAE y su traducción física operativa para termografía industrial.
    
    Parámetros:
        img_referencia: Captura térmica cruda del sensor.
        img_generada: Salida del modelo de difusión.
        sensibilidad_c_por_nivel: Estimación de grados °C por nivel de gris (por defecto 0.4 °C/nivel).
    """
    ref = asegurar_dimensiones(img_referencia)
    gen = asegurar_dimensiones(img_generada)

    ref_gray = np.array(ref.convert("L"), dtype=np.float32)
    gen_gray = np.array(gen.convert("L"), dtype=np.float32)

    # Métricas crudas
    mae_val = float(np.mean(np.abs(ref_gray - gen_gray)))
    ssim_val = float(ssim(ref_gray, gen_gray, data_range=255.0))

    # Interpretación física
    delta_t_aprox = mae_val * sensibilidad_c_por_nivel

    # Evaluación de geometría (SSIM)
    if ssim_val >= 0.85:
        explicacion_ssim = "Geometría intacta (límites y componentes nítidos)"
    elif ssim_val >= 0.75:
        explicacion_ssim = "Deformación geométrica leve en bordes finos"
    else:
        explicacion_ssim = "⚠️ Distorsión estructural severa detectada"

    # Evaluación térmica (MAE / Delta T)
    if delta_t_aprox <= 1.5:
        explicacion_mae = f"Variación leve (~{delta_t_aprox:.1f} °C). Isotermas estables."
    elif delta_t_aprox <= 3.0:
        explicacion_mae = f"Desvío moderado (~{delta_t_aprox:.1f} °C). Posible falso gradiente."
    else:
        explicacion_mae = f"⚠️ Desvío crítico (~{delta_t_aprox:.1f} °C). Alucinación térmica severa."

    # Dictamen operativo
    apto_metrologia = (ssim_val >= 0.80) and (delta_t_aprox <= 2.5)
    veredicto = "✅ Apto para reporte visual ilustrativo" if apto_metrologia else "⚠️ Riesgo de falsa anomalía térmica"

    return {
        "ssim": round(ssim_val, 4),
        "mae": round(mae_val, 2),
        "delta_t_c": round(delta_t_aprox, 2),
        "explicacion_ssim": explicacion_ssim,
        "explicacion_mae": explicacion_mae,
        "veredicto": veredicto,
        "apto": apto_metrologia
    }