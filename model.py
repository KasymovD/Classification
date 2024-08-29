import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(F1, (1, 1), strides=(1, 1), name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2c')(X)

    X = layers.add([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(name=bn_name_base + '2c')(X)

    X_shortcut = layers.Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1')(X_shortcut)
    X_shortcut = layers.BatchNormalization(name=bn_name_base + '1')(X_shortcut)

    X = layers.add([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = layers.Input(input_shape)

    X = layers.ZeroPadding2D((3, 3))(X_input)

    X = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = layers.BatchNormalization(name='bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    X = layers.Flatten()(X)
    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

# Загрузка данных из `Processed_Dataset/公司` и `Processed_Dataset/關防-整理好的`
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

# Загрузка данных
images_1, labels_1 = load_images_from_directory('Processed_Dataset/公司')
images_2, labels_2 = load_images_from_directory('Processed_Dataset/關防-整理好的')

# Проверка форм массивов
print(f"Shape of images_1: {images_1.shape}")
print(f"Shape of images_2: {images_2.shape}")

if images_1.ndim == images_2.ndim and len(images_2) > 0:
    images = np.concatenate([images_1, images_2], axis=0)
    labels = np.concatenate([labels_1, labels_2], axis=0)
else:
    raise ValueError("Dimension mismatch between datasets")

# Нормализация изображений
images = images / 255.0

# Преобразование меток в числовые значения и сохранение classes.pkl
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = tf.keras.utils.to_categorical(labels_encoded)

# Сохранение классов в файл classes.pkl
joblib.dump(label_encoder.classes_, 'classes.pkl')

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Создание модели ResNet50
model = ResNet50(input_shape=(64, 64, 3), classes=len(label_encoder.classes_))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=16, validation_data=(X_test, y_test))

# Сохранение модели
model.save("resnet50_custom_model.h5")