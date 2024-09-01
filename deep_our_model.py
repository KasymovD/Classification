import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

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

def residual_block(X, filters, kernel_size=3, stride=1):
    shortcut = X

    X = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(X)
    X = layers.BatchNormalization()(X)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    X = layers.add([X, shortcut])
    X = layers.Activation('relu')(X)

    return X

images_1, labels_1 = load_images_from_directory('Processed_Dataset/公司')
images_2, labels_2 = load_images_from_directory('Processed_Dataset/關防-整理好的')

if images_1.ndim == images_2.ndim and len(images_2) > 0:
    images = np.concatenate([images_1, images_2], axis=0)
    labels = np.concatenate([labels_1, labels_2], axis=0)
else:
    raise ValueError("Dimension mismatch between datasets")

images = images / 255.0

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = tf.keras.utils.to_categorical(labels_encoded)

joblib.dump(label_encoder.classes_, 'classes.pkl')

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

input_shape = (64, 64, 3)
inputs = layers.Input(shape=input_shape)

X = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
X = layers.BatchNormalization()(X)
X = layers.Activation('relu')(X)
X = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)

for i in range(3):
    X = residual_block(X, 64)
for i in range(4):
    X = residual_block(X, 128, stride=2)
for i in range(6):
    X = residual_block(X, 256, stride=2)
for i in range(3):
    X = residual_block(X, 512, stride=2)

X = layers.GlobalAveragePooling2D()(X)
X = layers.Dense(1024, activation='relu')(X)
X = layers.Dropout(0.5)(X)
outputs = layers.Dense(len(np.unique(labels)), activation='softmax')(X)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=45, validation_data=(X_test, y_test))

model.save("deep_resnet_model.h5")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 2])
plt.legend(loc='upper right')
plt.show()
