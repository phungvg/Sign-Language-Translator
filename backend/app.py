from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import base64
import numpy as np
import cv2

from handedness import handle_frame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(image: str = Form(...)):
    try:
        # decode base64 image
        image_bytes = base64.b64decode(image.split(';base64,')[1])

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = handle_frame(frame)

        if not result:
            return {"prediction": None}

        return {
            "prediction": {
                "landmarks": result["landmarks"],
                "label": str(result["label"]),
                "type": result["type"],
                "hand": result["hand"],
                "confidence": float(result["confidence"]),
            }
        }

    except Exception as e:
        print(e)
        return {"error": str(e)}