import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from PIL import Image
import tensorflow as tf
import joblib
from tensorflow.keras.metrics import top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# Функция загрузки изображений с учетом подклассов
def load_images(folder, img_size=(128, 128)):
    images = []
    labels = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                try:
                    pil_image = Image.open(img_path).convert('L').resize(img_size)
                    img = np.array(pil_image)
                    # Преобразуем в RGB, повторяя каналы
                    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
                    images.append(img)
                    # Создаем метку на основе относительного пути от корневой папки
                    relative_path = os.path.relpath(root, folder)
                    label = relative_path.replace(os.sep, '/')
                    labels.append(label)
                except Exception as e:
                    print(f"Ошибка обработки изображения {img_path}: {e}")

    images = np.array(images)
    return images, labels

# Функция создания residual block
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

# Загрузка изображений
print("Начинается загрузка данных...")
processed_data_folder = r'Processed_Dataset_sort'

images, labels = load_images(processed_data_folder)

# Нормализация изображений
images = images.astype('float32') / 255.0

# Кодирование меток
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Сохранение LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Сохранение классов (необязательно, если сохраняем LabelEncoder)
# joblib.dump(label_encoder.classes_, 'classes.pkl')

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Создание модели с residual block
input_shape = (128, 128, 3)
inputs = layers.Input(shape=input_shape)

X = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
X = layers.BatchNormalization()(X)
X = layers.Activation('relu')(X)
X = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)

# Строим несколько residual блоков
for i in range(3):
    X = residual_block(X, 64)
for i in range(4):
    X = residual_block(X, 128, stride=2)
for i in range(6):
    X = residual_block(X, 256, stride=2)
for i in range(3):
    X = residual_block(X, 512, stride=2)

# Завершаем архитектуру
X = layers.GlobalAveragePooling2D()(X)
X = layers.Dense(1024, activation='relu')(X)
X = layers.Dropout(0.5)(X)
outputs = layers.Dense(len(label_encoder.classes_), activation='softmax')(X)

# Создание модели
model = models.Model(inputs=inputs, outputs=outputs)

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', top_5_accuracy])

# Настройки ранней остановки и снижения learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Обучение модели
history = model.fit(X_train, y_train, epochs=45, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# Оценка модели
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Top-1 Accuracy: {scores[1] * 100:.2f}%")
print(f"Top-5 Accuracy: {scores[2] * 100:.2f}%")

# Сохранение модели
model.save('model.h5')

# Опционально: сохранение истории обучения для последующего анализа
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
