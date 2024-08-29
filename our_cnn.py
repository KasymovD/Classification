import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


def load_images_from_directory(directory, img_size=(64, 64)):
    images = []
    labels = []

    for root, dirs, files in os.walk(directory):
        class_name = os.path.basename(root)
        for file_name in files:
            img_path = os.path.join(root, file_name)
            if os.path.isfile(img_path):
                try:
                    img = Image.open(img_path).resize(img_size)
                    img = np.array(img)
                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Warning: Could not process image {img_path}. Error: {e}")

    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images from {directory}")
    return images, labels


images_1, labels_1 = load_images_from_directory('Processed_Dataset/公司')
images_2, labels_2 = load_images_from_directory('Processed_Dataset/關防-整理好的')

# Проверка форм массивов
print(f"Shape of images_1: {images_1.shape}")
print(f"Shape of images_2: {images_2.shape}")

if images_1.ndim == images_2.ndim and len(images_2) > 0:
    images = np.concatenate([images_1, images_2], axis=0)
    labels = np.concatenate([labels_1, labels_2], axis=0)
else:
    print(f"Dimension mismatch: images_1 has {images_1.ndim} dimensions, images_2 has {images_2.ndim} dimensions.")
    raise ValueError("Dimension mismatch between datasets")

images = images / 255.0

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = tf.keras.utils.to_categorical(labels_encoded)

joblib.dump(label_encoder, 'classes.pkl')

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

classifier = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(labels)), activation="softmax")
])

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
classifier.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test))
classifier.save("cnn_model_rgb.h5")