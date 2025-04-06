from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner, PILImage
from io import BytesIO
import sys

# 👇 fix PosixPath issue on Windows
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
app = FastAPI()

# Allow requests from Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your exported model
oracle = load_learner("oracle.pkl")

EDIBLE_CLASSES = [
    "shiitake", "enoki", "oyster", "chanterelle", "porcini", "morel", "button"
]

@app.post("/oracle")
async def classify_mushroom(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = PILImage.create(BytesIO(image_data))
        pred_class, pred_idx, probs = oracle.predict(img)

        return {
            "name": str(pred_class),
            "confidence": float(probs[pred_idx]),
            "isEdible": pred_class.lower() in (name.lower() for name in EDIBLE_CLASSES),
            "infoUrl": f"https://en.wikipedia.org/wiki/{str(pred_class).replace(' ', '_')}"
        }
    except Exception as e:
        print(f"Oracle threw error: {e}")
        return {
            "name": "unknown",
            "confidence": 0.0,
            "isEdible": False,
            "infoUrl": None
        }