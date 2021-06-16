from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


model = load_model('models/age_model.h5')


def age_recognition_func(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch /= 255.
    prediction = model.predict(img_batch)
    return prediction[0][0].round().astype('int')
