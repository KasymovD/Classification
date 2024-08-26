import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from PIL import Image
import tensorflow as tf
import joblib

def load_images(folder, img_size=(128, 128)):
    images = []
    labels = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                try:
                    pil_image = Image.open(img_path).convert('L')
                    pil_image = pil_image.resize(img_size)
                    img = np.array(pil_image)
                    images.append(img)

                    label = os.path.basename(root)
                    labels.append(label)
                except Exception as e:
                    print(f"Warning: Could not open or process the image {img_path}. Error: {e}")

    images = np.array(images)
    images = np.repeat(images[..., np.newaxis], 3, axis=-1)  # Повторяем канал для RGB
    return images, np.array(labels)

print("Starting data loading...")
processed_data_folder_1 = r'Processed_Dataset/公司'
processed_data_folder_2 = r'Processed_Dataset/關防-整理好的'

images_1, labels_1 = load_images(processed_data_folder_1)
print(f"Loaded {len(images_1)} images from {processed_data_folder_1}")

images_2, labels_2 = load_images(processed_data_folder_2)
print(f"Loaded {len(images_2)} images from {processed_data_folder_2}")

images = np.concatenate([images_1, images_2], axis=0)
labels = np.concatenate([labels_1, labels_2], axis=0)
print(f"Total images loaded: {len(images)}")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)
print("Labels encoding complete.")

joblib.dump(label_encoder.classes_, 'classes.pkl')
print("Class labels saved to 'classes.pkl'.")

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
print("Data split into training and test sets.")

print("Loading ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
print("ResNet50 model loaded.")

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
print("Model architecture created.")

for layer in base_model.layers[:-10]:
    layer.trainable = False
print("Base model layers frozen, except the last 10 layers.")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=1,
    restore_best_weights=True,
    mode='min'
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

print("Starting model training...")
model.fit(
    X_train, y_train,
    epochs=7,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print("Model training complete.")

model.save('stamp_classification_model_resnet50.h5')
print("Model saved as 'stamp_classification_model_resnet50.h5'.")
