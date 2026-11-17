import glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Importamos directamente tu función ya testeada
from src.diffusion import restaurar_con_modelo


def evaluar_experimento(ruta_imagen: str, intensidad: float = 0.25):
    print("=" * 65)
    print("🔬 INICIANDO ENSAYO METROLÓGICO INDEPENDIENTE")
    print(f"Archivo de prueba: {ruta_imagen}")
    print(f"Intensidad (Strength): {intensidad}")
    print("=" * 65)

    # 1. Cargar imagen original del sensor Testo
    img_orig = Image.open(ruta_imagen).convert("RGB")
    ancho_orig, alto_orig = img_orig.size
    print(f"Resolución nativa sensor: {ancho_orig}x{alto_orig} píxeles")

    # Referencia redimensionada a 512x512 para comparar en la misma escala
    img_ref = img_orig.resize((512, 512), Image.Resampling.LANCZOS)
    arr_ref = np.array(img_ref)

    # 2. Línea Base Tradicional (Interpolación Bicúbica clásica sin IA)
    img_bicubic = img_orig.resize((512, 512), Image.Resampling.BICUBIC)
    arr_bicubic = np.array(img_bicubic)

    # 3. Inferencia con Modelo A (SD v1.5)
    print("\n⏳ Procesando Modelo A (Stable Diffusion v1.5)...")
    img_a = restaurar_con_modelo(
        img_orig, "Stable Diffusion v1.5 (Estándar)", intensidad
    )
    arr_a = np.array(img_a)

    # 4. Inferencia con Modelo B (SD-Turbo)
    print("⏳ Procesando Modelo B (SD-Turbo)...")
    img_b = restaurar_con_modelo(
        img_orig, "SD-Turbo (Rápido / 1-4 pasos)", intensidad
    )
    arr_b = np.array(img_b)

    # 5. Cálculo de Métricas Objetivas (Fidelidad Estructural y Ruido)
    modelos = {
        "Bicúbica Clásica (Sin IA)": arr_bicubic,
        "Modelo A (SD v1.5)": arr_a,
        "Modelo B (SD-Turbo)": arr_b,
    }

    print("\n" + "-" * 65)
    print(f"{'MÉTODO':<28} | {'PSNR (dB)':<10} | {'SSIM (0-1)':<10} | {'MAE Píxeles'}")
    print("-" * 65)

    for nombre, arr_mod in modelos.items():
        # SSIM compara similitud estructural (1.0 = idéntico)
        valor_ssim = ssim(arr_ref, arr_mod, channel_axis=2)

        # PSNR mide relación señal a ruido (más alto = menos distorsión)
        valor_psnr = psnr(arr_ref, arr_mod)

        # MAE: Error Absoluto Medio entre los canales de intensidad térmica
        mae = np.mean(np.abs(arr_ref.astype(float) - arr_mod.astype(float)))

        print(f"{nombre:<28} | {valor_psnr:<10.2f} | {valor_ssim:<10.4f} | {mae:.2f}")

    print("-" * 65)
    print("📌 INTERPRETACIÓN RÁPIDA:")
    print("- SSIM cercano a 1.0 indica menor alteración de isotermas originales.")
    print("- Un MAE bajo certifica que la IA no modificó drásticamente los colores térmicos.")
    print("=" * 65)


if __name__ == "__main__":
    # Busca la primera imagen disponible en tu carpeta de ejemplos
    archivos = glob.glob("assets/ejemplos/*.*")
    imagenes = [
        a for a in archivos if a.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not imagenes:
        print("⚠️ No se encontraron imágenes en 'assets/ejemplos/'. Colocá una imagen para probar.")
    else:
        evaluar_experimento(imagenes[0], intensidad=0.25)