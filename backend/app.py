from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import base64
import numpy as np
import cv2

from handedness import handle_frame
from postprocess import suggest_completions
from spellchecker import SpellChecker

spell = SpellChecker()

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
                "landmarks": result.get("landmarks"),
                "label": str(result["label"]),
                "type": result.get("type"),
                "hand": result.get("hand"),
                "confidence": float(result["confidence"]) if result.get("confidence") is not None else 1.0,
            }
        }

    except Exception as e:
        print(e)
        return {"error": str(e)}


@app.post("/correct")
def correct(word: str = Form(...)):
    """Spell-correct a completed word and return top suggestions."""
    w = word.lower().strip()
    if not w:
        return {"corrected": "", "suggestions": []}

    corrected = spell.correction(w) or w
    suggestions = suggest_completions(w, n=5)

    return {"corrected": corrected, "suggestions": suggestions}


@app.post("/suggest")
def suggest(prefix: str = Form(...)):
    """Return live word completions for a partial word."""
    results = suggest_completions(prefix.lower().strip(), n=5)
    return {"suggestions": results}