import torch
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from PIL import Image

_modelos_cargados = {}
_dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _dispositivo == "cuda" else torch.float32


def obtener_pipeline(nombre_modelo: str):
    """Carga el pipeline seleccionado una sola vez y lo mantiene en memoria."""
    if nombre_modelo in _modelos_cargados:
        return _modelos_cargados[nombre_modelo]

    if "1.5" in nombre_modelo or "Estándar" in nombre_modelo:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=_dtype,
            safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if _dispositivo == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")

    elif "Turbo" in nombre_modelo:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=_dtype,
            safety_checker=None,
        )
        if _dispositivo == "cuda":
            pipe.to("cuda")
        else:
            pipe.to("cpu")
    else:
        raise ValueError(f"Modelo no reconocido: {nombre_modelo}")

    _modelos_cargados[nombre_modelo] = pipe
    return pipe


def restaurar_con_modelo(
    imagen: Image.Image, nombre_modelo: str, intensidad: float = 0.25
) -> Image.Image:
    """Ejecuta el modelo de difusión sobre la captura térmica.

    Garantiza dimensiones 512x512 y pasos suficientes para evitar colapso de
    tensores.
    """
    if imagen is None:
        return None

    # 1. Asegurar formato RGB y dimensiones exactas 512x512
    img_trabajo = imagen.convert("RGB")
    img_trabajo = img_trabajo.resize((512, 512), Image.Resampling.LANCZOS)

    pipe = obtener_pipeline(nombre_modelo)

    prompt = (
        "sharp infrared thermography image, clean industrial radiometric palette, "
        "precise temperature contours, clear borders, thermal inspection"
    )
    negative_prompt = (
        "blurry, optical photograph, low quality, artifacts, distorted"
        " temperature gradients, noise"
    )

    intensidad_val = max(0.05, min(float(intensidad), 0.95))

    # 2. Inferencia garantizando pasos mínimos para que el VAE/Scheduler no quede en 0
    if "Turbo" in nombre_modelo:
        # En img2img con SD-Turbo se recomiendan al menos 4-6 pasos totales para que con strength calcule >= 1 paso real
        total_steps = max(4, int(4 / intensidad_val))
        salida = pipe(
            prompt=prompt,
            image=img_trabajo,
            strength=intensidad_val,
            num_inference_steps=total_steps,
            guidance_scale=0.0,
        )
    else:
        # En SD v1.5 usamos 25 pasos base
        salida = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_trabajo,
            strength=intensidad_val,
            num_inference_steps=25,
            guidance_scale=6.0,
        )

    return salida.images[0]