# recognition/views.py
from django.http import JsonResponse
from .services import handle_frame
import base64
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2 

@csrf_exempt
def predict(request):
    try:
        if request.method == "POST":
            image = request.POST.get("image")

            try:
                image_bytes = base64.b64decode(image.split(';base64,')[1])

            except Exception:
                return JsonResponse({"error": "Invalid image format"}, status=400)

            # convert to numpy image
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            text = handle_frame(frame) # handle letter prediction

            return JsonResponse({"text": text}, status=200)
        return JsonResponse({"error": "POST method only"}, status=405)
    except Exception as e:
        print(e)
        return JsonResponse({"error": str(e)}, status=500)