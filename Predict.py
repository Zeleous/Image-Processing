import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model




def preprocess_image(img_path, target_size = (224, 224)):
    img = image.load_img(img_path, target_size=target_size, color_mode='rgb')
    img_array=image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_glasses(img_path):
    preP_image = preprocess_image(img_path)
    prediction = model.predict(preP_image)
    probability = prediction[0][0]
    
    if probability < 0.5:
        return "wearing glasses"
    else:
        return  "Not wearing glasses"
model = load_model('glasses_detection_model.h5')

image_path = 'glasses_detection/images/Zack.png'
result = predict_glasses(image_path)
print(f"Prediction: {result}")