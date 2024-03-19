from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import os
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np

# Create your views here.
# Load the pre-trained model
model_path = os.path.join(settings.BASE_DIR, 'BrainTumor10EpochsCategorical.h5')
model = load_model(model_path)

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image_file.name)
        with open(image_path, 'wb') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        # Load and preprocess the image for prediction
        image = cv2.imread(image_path)
        img = Image.fromarray(image)
        img = img.resize((64,64))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        
        # Make prediction
        result = np.argmax(model.predict(input_img), axis=-1)
        if result == 0:
            prediction = "No Tumor"
        else:
            prediction = "Tumor Detected"
        
        return JsonResponse({'prediction': prediction})
    
    return render(request, 'detection/index.html')
