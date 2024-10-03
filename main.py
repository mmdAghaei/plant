import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

model = load_model('model/finalPlant2.h5')

with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)


# for index, class_name in class_indices.items():
#     print(f"Class Name: {class_name}")

def predict_image_class(model, img_path, class_indices):
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    
  
    predictions = model.predict(img_array)
    predicted_class_index = str(np.argmax(predictions, axis=1)[0])  
    

    if predicted_class_index in class_indices:
        class_name = class_indices[predicted_class_index]
    
    return class_name

## آدرس عکس
image_path = 'image/image.png' 

predicted_class_name = predict_image_class(model, image_path, class_indices)
print("------------------------------------")
print("Predicted:", predicted_class_name.replace('___', ' ').replace('_', ' '))
print("------------------------------------")

