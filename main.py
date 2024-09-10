import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from main_ui import Ui_MainWindow
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
import cv2
from PIL import ImageEnhance
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from tensorflow.keras.optimizers import Adam
import joblib
import numpy as np
import datetime
import random
import glob


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = load_model('stamp_classification_model_resnet50.h5')
classes = joblib.load('classes.pkl')
label_encoder = LabelEncoder()
label_encoder.fit(classes)


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
                    print(f"Error loading image {file}: {e}")

    images = np.array(images)
    images = np.repeat(images[..., np.newaxis], 3, axis=-1)
    return images, np.array(labels)


class TrainingThread(QThread):
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)

    def __init__(self, image_folders, model_path, classes_path):
        super().__init__()
        self.image_folders = image_folders
        self.model_path = model_path
        self.classes_path = classes_path

    def load_images(self, folder, img_size=(128, 128)):
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
                        print(f"載入圖片時出錯 {file}: {e}") # Error loading image
        images = np.array(images)
        images = np.repeat(images[..., np.newaxis], 3, axis=-1)
        return images, np.array(labels)

    def run(self):
        try:
            self.update_status.emit("正在載入模型和類別...") #Загрузка модели и классов...
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
            label_encoder = LabelEncoder()
            classes = joblib.load(self.classes_path)
            label_encoder.fit(classes)

            default_folder_1 = r'Processed_Dataset/公司'
            default_folder_2 = r'Processed_Dataset/關防-整理好的'

            images_1, labels_1 = self.load_images(default_folder_1)
            images_2, labels_2 = self.load_images(default_folder_2)

            images = np.concatenate([images_1, images_2], axis=0)
            labels = np.concatenate([labels_1, labels_2], axis=0)

            for folder in self.image_folders:
                new_images, new_labels = self.load_images(folder)
                images = np.concatenate([images, new_images], axis=0)
                labels = np.concatenate([labels, new_labels], axis=0)

            # Кодирование меток
            labels_encoded = label_encoder.fit_transform(labels)
            labels_encoded = to_categorical(labels_encoded)

            # Генерация имени для новых файлов с датой и временем
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"stamp_classification_model_resnet50_{current_time}.h5"
            classes_name = f"classes_{current_time}.pkl"

            joblib.dump(label_encoder.classes_, classes_name)
            self.update_status.emit(f"類別已保存到'{classes_name}'.") # Классы сохранены в

            X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
            self.update_status.emit("數據已分為訓練集和測試集") # Данные разделены на тренировочные и тестовые наборы.

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)

            for layer in base_model.layers[:-10]:
                layer.trainable = False

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

            self.update_status.emit("開始訓練模型...") # Начало обучения модели...
            epochs = 1
            for epoch in range(1, epochs + 1):
                model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=1)
                progress = int((epoch / epochs) * 100)
                self.update_progress.emit(progress)

            model.save(model_name)
            self.update_status.emit(f"訓練完成。模型已保存為 '{model_name}'.") # Обучение завершено. Модель сохранена как

        except Exception as e:
            self.update_status.emit(f"訓練過程中出現錯誤: {str(e)}") # Ошибка во время обучения


# def predict_stamp(img_path):
#     if model is None or label_encoder is None:
#         return "Модель или классы не загружены", None
#
#     img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
#     img_array = img_to_array(img)
#     img_array = np.repeat(img_array, 3, axis=-1)
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
#     predicted_label = label_encoder.inverse_transform(predicted_class)
#     return predicted_label[0], img_array

class PredictionThread(QThread):
    prediction_done = pyqtSignal(str, np.ndarray, str)
    update_progress = pyqtSignal(int)

    def __init__(self, img_path, model, label_encoder):
        super().__init__()
        self.img_path = img_path
        self.model = model
        self.label_encoder = label_encoder
    def run(self):

        self.update_progress.emit(50)
        predicted_label, img_array = predict_stamp(self.img_path, self.model, self.label_encoder)
        self.update_progress.emit(100)
        self.prediction_done.emit(predicted_label, img_array, self.img_path)
        predicted_label, img_array = predict_stamp(self.img_path, self.model, self.label_encoder)
        self.prediction_done.emit(predicted_label, img_array, self.img_path)



def predict_stamp(img_path, model, label_encoder):
    if model is None or label_encoder is None:
        return "模型或類別未載入", None # Модель или классы не загружены

    img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0], img_array

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Start_button.clicked.connect(self.start_prediction)
        self.ui.Model_choose_button.clicked.connect(self.choose_model)
        self.ui.Save_button.clicked.connect(self.choose_classes)
        self.ui.Clear_button.clicked.connect(self.clear_labels)
        self.ui.Crop_button.clicked.connect(self.prepare_stamps)
        self.ui.Generator_button.clicked.connect(self.start_generation)
        self.ui.Learn_button.clicked.connect(self.start_training)
        self.ui.Test_button.clicked.connect(self.start_testing)

        self.model_loaded = False
        self.classes_loaded = False

        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setMaximum(100)

    def start_testing(self):
        try:
            model_path, _ = QFileDialog.getOpenFileName(self, '請選擇要測試的模型', '', 'H5 files (*.h5)') #Выберите модель для тестирования
            if not model_path:
                self.ui.analyz.setText("錯誤：未選擇模型！") # Ошибка: модель не выбрана
                return
            self.ui.analyz.setText(f"模型已選擇: {model_path}") # Выбрана модель

            model = load_model(model_path)

            classes_path, _ = QFileDialog.getOpenFileName(self, '請選擇類別文件', '', 'Pickle files (*.pkl)') #Выберите файл классов
            if not classes_path:
                self.ui.analyz.setText("錯誤：未選擇類別文件！") #Ошибка: файл классов не выбран!
                return
            self.ui.analyz.setText(f"類別文件已選擇: {classes_path}") #Выбран файл классов

            label_encoder = LabelEncoder()
            classes = joblib.load(classes_path)
            label_encoder.fit(classes)

            base_folders = [r'Processed_Dataset/公司', r'Processed_Dataset/關防-整理好的']
            all_classes = []
            for folder in base_folders:
                all_classes.extend([os.path.join(folder, subfolder) for subfolder in os.listdir(folder) if
                                    os.path.isdir(os.path.join(folder, subfolder))])

            if len(all_classes) < 5:
                self.ui.analyz.setText("錯誤：測試的類別數量不足！")
                return

            random_classes = random.sample(all_classes, 5)
            test_images = {}
            for class_folder in random_classes:
                image_paths = glob.glob(os.path.join(class_folder, '*.png')) + \
                              glob.glob(os.path.join(class_folder, '*.jpg')) + \
                              glob.glob(os.path.join(class_folder, '*.jpeg'))
                if len(image_paths) > 50:
                    image_paths = random.sample(image_paths, 50)
                test_images[class_folder] = image_paths

            total_accuracy = 0
            num_classes = len(test_images)

            for class_folder, images in test_images.items():
                correct_predictions = 0
                for _ in range(3):
                    for img_path in images:
                        predicted_label, _ = predict_stamp(img_path, model,
                                                           label_encoder)
                        true_label = os.path.basename(class_folder)

                        if predicted_label == true_label:
                            correct_predictions += 1

                accuracy = (correct_predictions / (3 * len(images))) * 100
                total_accuracy += accuracy
                self.ui.analyz.setText(f"的準確率 {class_folder}: {accuracy:.2f}%") #Точность для

            average_accuracy = total_accuracy / num_classes
            self.ui.analyz.setText(f"所有類別的平均準確率: {average_accuracy:.2f}%") #Средняя точность по всем классам

        except Exception as e:
            self.ui.analyz.setText(f"錯誤: {str(e)}")

    def start_training(self):
        try:
            model_path, _ = QFileDialog.getOpenFileName(self, '選擇模型', '', 'H5 files (*.h5)') #
            if not model_path:
                self.ui.analyz.setText("錯誤：未選擇類別文件！") # Ошибка: модель не выбрана!
                return

            classes_path, _ = QFileDialog.getOpenFileName(self, '請選擇類別文件', '', 'Pickle files (*.pkl)') # Выберите файл классов
            if not classes_path:
                self.ui.analyz.setText("Ошибка: файл классов не выбран!") # Ошибка: файл классов не выбран!
                return

            folders = []
            while True:
                folder = QFileDialog.getExistingDirectory(self,
                                                          '請選擇包含圖片的文件夾（按取消以結束') #Выберите папку с изображениями (нажмите Отмена для завершения)
                if folder:
                    folders.append(folder)
                    self.ui.analyz.setText(f"已添加文件夾: {folder}") # Добавлена папка
                else:
                    break

            if not folders:
                self.ui.analyz.setText("錯誤：未選擇任何文件夾！") # Ошибка: ни одна папка не выбрана!
                return

            self.training_thread = TrainingThread(folders, model_path, classes_path)
            self.training_thread.update_progress.connect(self.update_progress_bar)
            self.training_thread.update_status.connect(self.ui.analyz.setText)
            self.training_thread.start()

        except Exception as e:
            self.ui.analyz.setText(f"錯誤: {str(e)}") #Ошибка

    def update_progress_bar(self, value):
        self.ui.progressBar.setValue(value)


    def load_images_from_folder(self, folder, img_size=(128, 128)):
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
        images = np.repeat(images[..., np.newaxis], 3, axis=-1)
        return images, np.array(labels)

    def start_generation(self):
        file_dialog = QFileDialog()
        img_paths, _ = file_dialog.getOpenFileNames(self, '選擇圖片', '',
                                                    'Images (*.png *.jpg *.jpeg *.bmp)')

        if img_paths:
            output_folder = QFileDialog.getExistingDirectory(self, '選擇保存圖片的文件夾') # Выбрать папку для сохранения изображений

            if output_folder:
                for img_path in img_paths:
                    self.augment_image_and_save(img_path, output_folder)

                self.ui.analyz.setText(
                    f'圖片的第 {len(img_paths)} 次增強已完成。已保存到 {output_folder}.') # Аугментация 2 изображений завершена. Сохранены в

    def augment_image_and_save(self, img_path, output_folder, augment_count=30):
        image = self.load_image(img_path)

        if image:
            base_filename = os.path.splitext(os.path.basename(img_path))[0]

            for i in range(augment_count):
                augmented_image = self.augment_image(image)
                augmented_filename = f"{base_filename}_augmentation_{i + 1}.png"
                augmented_image.save(os.path.join(output_folder, augmented_filename))

            print(f"圖片增強 {img_path} 已完成並保存到 {output_folder}") # Аугментация изображения   завершена и сохранена в
        else:
            print(f"無法載入圖片 {img_path}") #"Не удалось загрузить изображение"

    def load_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((128, 128))
            return image
        except Exception as e:
            print(f"載入圖片時出錯 {image_path}: {e}")
            return None

    def augment_image(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(np.random.uniform(0.8, 1.2))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(np.random.uniform(0.8, 1.2))

        image_array = np.array(image)
        noise = np.random.normal(0, 3, image_array.shape).astype(np.uint8)
        image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

        image = image.rotate(np.random.uniform(0, 360))

        return image

    def prepare_stamps(self):
        file_dialog = QFileDialog()
        img_paths, _ = file_dialog.getOpenFileNames(self, '選擇圖片', '', 'Images (*.png *.xpm *.jpg *.bmp)')

        if img_paths:
            output_folder = QFileDialog.getExistingDirectory(self, '選擇用於保存圖片的輸出資料夾') # Выбрать выходную папку для сохранения изображений

            if output_folder:
                for img_path in img_paths:
                    self.prepare_and_save_image(img_path, output_folder)

                self.ui.analyz.setText(f'已準備好 {len(img_paths)} 圖片已保存到 {output_folder}.') #Подготовлено изображений и сохранено в

    def prepare_and_save_image(self, img_path, output_folder, img_size=(128, 128)):
        try:
            pil_image = Image.open(img_path).convert('L')
            pil_image = pil_image.resize(img_size)
            img = np.array(pil_image)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            output_path = os.path.join(output_folder, os.path.basename(img_path))
            pil_image.save(output_path)

        except Exception as e:
            print(f"警告：無法開啟或處理圖片 {img_path}. 錯誤: {e}") #Warning: Could not open or process the image. Error

    def choose_model(self):
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(self, '選擇模型', '', 'H5 files (*.h5)') #Выбрать модель

        if model_path:
            global model
            model = load_model(model_path)
            self.model_loaded = True
            self.update_analyz()

    def choose_classes(self):
        file_dialog = QFileDialog()
        classes_path, _ = file_dialog.getOpenFileName(self, '選擇類別文件', '', 'Pickle files (*.pkl)') #Выбрать файл классов

        if classes_path:
            global label_encoder
            classes = joblib.load(classes_path)
            label_encoder.fit(classes)
            self.classes_loaded = True
            self.update_analyz()

    def update_analyz(self):
        if self.model_loaded and self.classes_loaded:
            self.ui.analyz.setText('模型和類別已載入。可以開始預測了') #'Модель и классы загружены. Можно начинать предсказание.'
        elif self.model_loaded:
            self.ui.analyz.setText('模型已載入。請選擇類別文件') #'Модель загружена. Пожалуйста, выберите файл классов.'
        elif self.classes_loaded:
            self.ui.analyz.setText('類別已載入。請選擇模型') #'Классы загружены. Пожалуйста, выберите модель.'
        else:
            self.ui.analyz.setText('请先选择模型和类别文件') #'Классы загружены. Пожалуйста, выберите модель.'

    def start_prediction(self):

        if not self.model_loaded or not self.classes_loaded:
            self.ui.analyz.setText('錯誤：模型或類別未選擇！') #Ошибка: модель или классы не выбраны!
            return

        file_dialog = QFileDialog()
        img_path, _ = file_dialog.getOpenFileName(self, '選擇圖片', '', 'Images (*.png *.xpm *.jpg *.bmp)') #Выбрать изображение

        if img_path:
            self.prediction_thread = PredictionThread(img_path, model, label_encoder)
            self.prediction_thread.update_progress.connect(self.update_progress_bar)
            self.prediction_thread.prediction_done.connect(self.on_prediction_done)
            self.ui.progressBar.setValue(0)
            self.prediction_thread.start()

    def on_prediction_done(self, predicted_label, img_array, img_path):
        self.ui.input_image.setPixmap(QPixmap(img_path).scaled(self.ui.input_image.size(), Qt.KeepAspectRatio))
        if img_array is not None:
            feature_map_img = self.visualize_specific_feature_map(img_array, 'conv1_conv', 15)
            self.display_image_in_label(feature_map_img, self.ui.image2)
        self.ui.analyz.setText(f'這枚郵票屬於: {predicted_label}')

    def visualize_specific_feature_map(self, img_array, layer_name, feature_index):
        if model is None:
            return None

        layer = model.get_layer(name=layer_name)
        feature_model = Model(inputs=model.input, outputs=layer.output)
        feature_maps = feature_model.predict(img_array)

        feature_map = feature_maps[0, :, :, feature_index]
        feature_map_resized = cv2.resize(feature_map, (128, 128))
        return feature_map_resized

    def display_image_in_label(self, img_array, label):
        if img_array is None:
            return

        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
        img_array = img_array.astype(np.uint8)

        height, width = img_array.shape
        q_img = QImage(img_array.data, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))

    def clear_labels(self):
        self.ui.input_image.setText("印章圖像")
        self.ui.image2.setText("模型印章")
        self.ui.analyz.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
