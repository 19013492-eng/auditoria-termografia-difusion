import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

ETIQUETAS_CLASES = [
    "Termografia_Valida_OK",
    "Termografia_Descalibrada_Saturada",
    "Termografia_Fuera_Dominio"
]

transformacion_inferencia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def cargar_clasificador(ruta_pesos: str = "modelo_mobilenet_termografia.pth"):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = models.mobilenet_v2(weights=None)
    modelo.classifier[1] = nn.Linear(modelo.last_channel, 3)
    
    try:
        modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo))
    except Exception:
        pass
        
    modelo.to(dispositivo)
    modelo.eval()
    return modelo, dispositivo

modelo_clasificador, dispositivo = cargar_clasificador()

def auditar_calidad_termica(imagen: Image.Image):
    """
    Evalúa si la termografía restaurada mantiene la coherencia cromática
    o si la difusión introdujo saturación y artefactos.
    """
    if imagen is None:
        return "Sin imagen", 0.0

    tensor = transformacion_inferencia(imagen.convert("RGB")).unsqueeze(0).to(dispositivo)
    
    with torch.no_grad():
        salidas = modelo_clasificador(tensor)
        probabilidades = torch.softmax(salidas, dim=1)[0]
    
    indice_ganador = torch.argmax(probabilidades).item()
    clase = ETIQUETAS_CLASES[indice_ganador]
    confianza = probabilidades[indice_ganador].item() * 100.0

    if confianza < 70.0:
        return f"Dudosa / Revisión Manual ({clase})", confianza

    return clase, confianza