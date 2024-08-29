import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import joblib

model = load_model('cnn_model_rgb.h5')

classes = joblib.load('classes.pkl')

if isinstance(classes, LabelEncoder):
    classes = classes.classes_

if not isinstance(classes, np.ndarray):
    classes = np.array(classes)

label_encoder = LabelEncoder()
label_encoder.classes_ = classes

def predict_stamp(img_path, model, label_encoder):
    img = load_img(img_path, target_size=(64, 64), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

test_image_path = 'Processed_Dataset/公司/達正投資/IS212_augmentation_1.png'
predicted_label = predict_stamp(test_image_path, model, label_encoder)
print(f'This stamp belongs to: {predicted_label}')
