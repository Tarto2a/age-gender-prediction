from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
IMAGE_SIZE = 224
NUM_AGE_CLASSES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(pretrained=True)
        self.base.fc = nn.Identity()
        self.gender_head = nn.Linear(2048, 2)
        self.age_head = nn.Linear(2048, NUM_AGE_CLASSES)

    def forward(self, x):
        x = self.base(x)
        return self.gender_head(x), self.age_head(x)

# Load model
model = AgeGenderModel().to(DEVICE)
model.load_state_dict(torch.load("age_gender_resnet50.pth", map_location=DEVICE))
model.eval()

# Face detector (MTCNN)
mtcnn = MTCNN(keep_all=False, device=DEVICE)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Convert class to readable age range
def class_to_age_range(age_class):
    return "95+" if age_class == 19 else f"{age_class * 5}-{age_class * 5 + 4}"

# -----------------------
# Prediction Route
# -----------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image"}

    # Detect face
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return {"error": "No face detected"}

    # Use the first detected face
    x1, y1, x2, y2 = map(int, boxes[0])
    face = img.crop((x1, y1, x2, y2))

    input_tensor = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        gender_out, age_out = model(input_tensor)
        gender = torch.argmax(gender_out, 1).item()
        age_class = torch.argmax(age_out, 1).item()

    return {
        "gender": "Male" if gender == 0 else "Female",
        "age_range": class_to_age_range(age_class),
        "box": {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1
        }
    }
