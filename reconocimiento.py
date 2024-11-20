import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
model.load_state_dict(torch.load('residuo_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformaciones de validación
data_transforms = {
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Predecir una imagen
img = Image.open('plastico1.jpg')
#img = Image.open('metal3.jpg')
img = data_transforms['validation'](img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img).squeeze()
    pred = torch.sigmoid(output).item()
    print(f"Probabilidad: {pred}")
    print(f"Predicción: {'metal' if pred > 0.5 else 'plastic'}")
