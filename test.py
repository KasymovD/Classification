import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

model = load_model('stamp_classification_model_resnet50.h5')
classes = joblib.load('classes.pkl')

label_encoder = LabelEncoder()
label_encoder.fit(classes)


def predict_stamp(img_path, model, label_encoder):
    img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0], img_array


def visualize_specific_feature_map(model, img_array, layer_name, feature_index):
    layer = model.get_layer(name=layer_name)
    feature_model = Model(inputs=model.input, outputs=layer.output)

    feature_maps = feature_model.predict(img_array)

    feature_map = feature_maps[0, :, :, feature_index]
    plt.figure(figsize=(5, 5))
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')
    plt.show()


test_image_path = 'Dataset/公司/正華投資/DS077.png'
predicted_label, img_array = predict_stamp(test_image_path, model, label_encoder)
print(f'This stamp belongs to: {predicted_label}')

visualize_specific_feature_map(model, img_array, 'conv1_conv', 8)  # Замените 5 на индекс вашего feature map


#all
# import numpy as np
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import matplotlib.pyplot as plt
#
# # Загрузка модели и классов
# model = load_model('stamp_classification_model_resnet50.h5')
#
# classes = joblib.load('classes.pkl')
#
# label_encoder = LabelEncoder()
# label_encoder.fit(classes)
#
#
# def predict_stamp(img_path, model, label_encoder):
#     img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
#     img_array = img_to_array(img)
#     img_array = np.repeat(img_array, 3, axis=-1)
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
#     predicted_label = label_encoder.inverse_transform(predicted_class)
#     return predicted_label[0], img_array
#
#
# def visualize_specific_feature_map(model, img_array, layer_name, feature_index):
#     layer = model.get_layer(name=layer_name)
#     feature_model = Model(inputs=model.input, outputs=layer.output)
#
#     feature_maps = feature_model.predict(img_array)
#
#     feature_map = feature_maps[0, :, :, feature_index]
#     plt.figure(figsize=(5, 5))
#     plt.imshow(feature_map, cmap='viridis')
#     plt.axis('off')
#     plt.show()
#
#
# test_image_path = 'Dataset/公司/正華投資/DS077.png'
# predicted_label, img_array = predict_stamp(test_image_path, model, label_encoder)
# print(f'This stamp belongs to: {predicted_label}')
#
# visualize_specific_feature_map(model, img_array, 'conv1_conv', 5)
