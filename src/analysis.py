import torch
from transformers import pipeline
from PIL import Image

_clasificador_zeroshot = None

def obtener_analizador():
    global _clasificador_zeroshot
    if _clasificador_zeroshot is None:
        dispositivo = 0 if torch.cuda.is_available() else -1
        _clasificador_zeroshot = pipeline(
            "zero-shot-image-classification",
            model="google/siglip-base-patch16-224",
            device=dispositivo
        )
    return _clasificador_zeroshot

def analizar_termografia_zeroshot(imagen: Image.Image) -> str:
    """
    Analiza la calidad visual cualitativa comparando antes y después
    mediante descripciones en lenguaje natural con SigLIP.
    """
    if imagen is None:
        return "No hay imagen para analizar."

    analizador = obtener_analizador()
    
    etiquetas = [
        "una captura termográfica industrial nítida y bien definida",
        "una imagen térmica con ruido visual o artefactos",
        "una termografía borrosa con pérdida de bordes",
        "una imagen saturada con exceso de brillo o contraste"
    ]
    
    salida = analizador(imagen.convert("RGB"), candidate_labels=etiquetas)
    
    lineas = ["### 🔍 Análisis Visual Cualitativo (SigLIP Zero-Shot)"]
    for item in salida[:2]:
        porcentaje = item["score"] * 100.0
        lineas.append(f"- **{item['label']}**: `{porcentaje:.1f}%`")
        
    return "\n".join(lineas)