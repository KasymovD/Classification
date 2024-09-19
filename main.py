import pickle
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication, QMainWindow
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from main_ui import Ui_MainWindow
import our_model
import os
import io
import subprocess
from PIL import Image, ImageEnhance
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
import sys
from utils import resource_path  # Импортируем из utils.py


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()

        self.model_path = None
        self.label_encoder_path = None
        self.gan_model_path = None
        self.generated_images_dir = None
        self.input_image_path = None

        self.ui.Model_choose_button.clicked.connect(self.choose_model)
        self.ui.Save_button.clicked.connect(self.choose_label_encoder)
        self.ui.Start_button.clicked.connect(self.start_process)
        self.ui.Clear_button.clicked.connect(self.clear_labels)
        self.ui.Generator_button.clicked.connect(self.start_generation)
        self.ui.Crop_button.clicked.connect(self.prepare_stamps)
        self.ui.Test_button.clicked.connect(self.open_image_dialog)
        self.ui.Learn_button.clicked.connect(self.history_of_train_choose)

        self.ui.progressBar.setValue(0)


    def initUI(self):
        self.setFixedSize(1920, 1350)

    def history_of_train_choose(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇歷史文件", "", "Pickle Files (*.pkl)")

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    self.history = pickle.load(f)
                self.history_of_train()
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"文件加載錯誤: {str(e)}")
        else:
            QMessageBox.information(self, "取消", "未選擇文件")

    def history_of_train(self):
        self.ui.progressBar.setValue(0)
        top1_accuracy = self.history.get('accuracy')
        top5_accuracy = self.history.get('top_5_accuracy')
        self.ui.analyz.setText("印章類別")
        self.ui.analyz_2.setText("相似度百分比")
        self.ui.progressBar.setValue(50)

        if top1_accuracy:
            max_top1 = "{:.2f}".format(max(top1_accuracy))
            self.ui.analyz_3.setText(f"Top-1 Accuracy: {max_top1}%")
            self.ui.progressBar.setValue(100)

        else:
            self.ui.analyz_3.setText("未找到 Top-1 準確率")
            self.ui.progressBar.setValue(100)


        if top5_accuracy:
            max_top5 = "{:.4f}".format(max(top5_accuracy))
            current_text = self.ui.analyz_3.text()
            self.ui.analyz_3.setText(f"{current_text}\nTop-5 Accuracy: {max_top5}%")
            self.ui.progressBar.setValue(100)

        else:
            current_text = self.ui.analyz_3.text()
            self.ui.analyz_3.setText(f"{current_text}\n未找到 Top-5 準確率")
            self.ui.progressBar.setValue(100)

    def open_image_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "選擇圖像", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp)", options=options)

        if file_name:
            self.extract_and_display_all_metadata(file_name)

    def extract_and_display_all_metadata(self, image_path):
        try:
            parser = createParser(image_path)
            if not parser:
                self.ui.analyz_3.setText("無法為文件創建解析器.")
                return

            metadata = extractMetadata(parser)

            if not metadata:
                self.ui.analyz_3.setText("未找到元數據.")
                return

            metadata_str = ""
            for line in metadata.exportPlaintext():
                metadata_str += line + "\n"

            self.ui.analyz_3.setText(metadata_str)

        except Exception as e:
            self.ui.analyz_3.setText(f"提取數據時發生錯誤: {str(e)}")

    def prepare_stamps(self):
        file_dialog = QFileDialog()
        img_paths, _ = file_dialog.getOpenFileNames(self, '選擇圖片', '', 'Images (*.png *.xpm *.jpg *.bmp)')

        if img_paths:
            output_folder = QFileDialog.getExistingDirectory(self,
                                                             '選擇用於保存圖片的輸出資料夾')

            if output_folder:
                for img_path in img_paths:
                    self.prepare_and_save_image(img_path, output_folder)

                self.ui.analyz.setText(
                    f'已準備好 {len(img_paths)} 圖片已保存到 {output_folder}.')

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
            print(
                f"警告：無法開啟或處理圖片 {img_path}. 錯誤: {e}")

    def start_generation(self):
        self.ui.progressBar.setValue(0)
        file_dialog = QFileDialog()
        img_paths, _ = file_dialog.getOpenFileNames(self, '選擇圖片', '',
                                                    'Images (*.png *.jpg *.jpeg *.bmp)')

        if img_paths:
            output_folder = QFileDialog.getExistingDirectory(self,
                                                             '選擇保存圖片的文件夾')

            if output_folder:
                for img_path in img_paths:
                    self.augment_image_and_save(img_path, output_folder)

                self.ui.analyz.setText(
                    f'圖片的第 \n{len(img_paths)} 次增強已完成。\n已保存到 {output_folder}.')

    def augment_image_and_save(self, img_path, output_folder, augment_count=30):
        image = self.load_image(img_path)
        self.ui.progressBar.setValue(50)

        if image:
            base_filename = os.path.splitext(os.path.basename(img_path))[0]

            for i in range(augment_count):
                augmented_image = self.augment_image(image)
                augmented_filename = f"{base_filename}_fake_{i + 1}.png"
                augmented_image.save(os.path.join(output_folder, augmented_filename))

            print(
                f"圖片增強 {img_path} 已完成並保存到 {output_folder}")
            self.ui.progressBar.setValue(100)

        else:
            print(f"無法載入圖片 {img_path}")

    def load_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((128, 128))
            return image
        except Exception as e:
            print(f"載入圖片時出錯 {image_path}: {e}")
            self.ui.progressBar.setValue(100)
            return None

    def augment_image(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(np.random.uniform(0.7, 1.1))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(np.random.uniform(0.7, 1.1))

        image_array = np.array(image)
        noise = np.random.normal(0, 0.5, image_array.shape).astype(np.uint8)
        image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

        image = image.rotate(np.random.uniform(0, 360))

        return image

    def choose_model(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇 CNN 模型", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if fileName:
            self.model_path = fileName
            QtWidgets.QMessageBox.information(
                self, "模型已選擇", f"您已選擇模型: {os.path.basename(fileName)}")

    def choose_label_encoder(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇類別文件", "", "Pickle Files (*.pkl);;All Files (*)", options=options)
        if fileName:
            self.label_encoder_path = fileName
            QtWidgets.QMessageBox.information(
                self, "類別文件已選擇", f"您已選擇類別文件: {os.path.basename(fileName)}")

    def start_process(self):
        if not self.model_path or not self.label_encoder_path:
            QtWidgets.QMessageBox.warning(
                self, "錯誤", "請在開始前選擇模型和類別文件。")
            return

        options = QtWidgets.QFileDialog.Options()
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇圖像", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if image_path:
            self.input_image_path = image_path
            self.ui.progressBar.setValue(0)
            QtWidgets.QApplication.processEvents()

            pixmap = QtGui.QPixmap(image_path)
            self.ui.input_image.setPixmap(
                pixmap.scaled(self.ui.input_image.size(), QtCore.Qt.KeepAspectRatio))
            self.ui.input_image.setAlignment(QtCore.Qt.AlignCenter)

            self.ui.progressBar.setValue(50)
            QtWidgets.QApplication.processEvents()

            self.process_image(image_path)

            self.ui.progressBar.setValue(100)

    def process_image(self, image_path):
        try:
            results = our_model.full_predict(
                image_path,
                self.model_path,
                self.label_encoder_path
            )

            main_category = results['main_category']
            sub_categories = results['sub_categories']
            cluster_name = results['cluster_name']
            similarity = results['similarity']
            most_similar_image = results['most_similar_image']
            features_original = results['features_original']
            features_similar = results['features_similar']
            features_image = results['features_image']
            features_similar_image = results['features_similar_image']

            sub_categories_str = ' -> '.join(sub_categories) if sub_categories else '無'
            if int(cluster_name) == 0:
                self.ui.analyz.setText(f"類別: {main_category}\n子類別: {sub_categories_str}\n集群/來源: A")
                self.ui.analyz_2.setText(f"相似度: {similarity:.2f}%")
            elif int(cluster_name) == 1:
                self.ui.analyz.setText(f"類別: {main_category}\n子類別: {sub_categories_str}\n集群/來源: B")
                self.ui.analyz_2.setText(f"相似度: {similarity:.2f}%")
            elif int(cluster_name) == 2:
                self.ui.analyz.setText(f"類別: {main_category}\n子類別: {sub_categories_str}\n集群/來源: C")
                self.ui.analyz_2.setText(f"相似度: {similarity:.2f}%")
            if similarity >= 85:
                self.ui.analyz_3.setText("找到相似的圖像✅")
            else:
                self.ui.analyz_3.setText("未找到相似的圖像❌")
                self.ui.analyz.setText(f"類別: {0}\n子類別: {0}\n集群/來源: 0")
                self.ui.analyz_2.setText(f"相似度: {similarity:.2f}%")

            features_pixmap = QtGui.QPixmap.fromImage(features_image)
            self.ui.image2.setPixmap(
                features_pixmap.scaled(self.ui.image2.size(), QtCore.Qt.KeepAspectRatio))
            self.ui.image2.setAlignment(QtCore.Qt.AlignCenter)

            if similarity >= 85 and most_similar_image:
                pixmap_similar = QtGui.QPixmap(most_similar_image)
                self.ui.model_image.setPixmap(
                    pixmap_similar.scaled(self.ui.model_image.size(), QtCore.Qt.KeepAspectRatio))
                self.ui.model_image.setAlignment(QtCore.Qt.AlignCenter)

                features_similar_pixmap = QtGui.QPixmap.fromImage(
                    features_similar_image)
                self.ui.label.setPixmap(
                    features_similar_pixmap.scaled(self.ui.label.size(), QtCore.Qt.KeepAspectRatio))
                self.ui.label.setAlignment(QtCore.Qt.AlignCenter)
            else:
                self.ui.model_image.clear()
                self.ui.model_image.setText(
                    "與原始印章最相似的照片")

                self.ui.label.clear()
                self.ui.label.setText("原始印章的特徵")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "錯誤", f"處理圖像時發生錯誤:\n{e}")

    def clear_labels(self):
        self.ui.input_image.clear()
        self.ui.input_image.setText("原始印章")

        self.ui.image2.clear()
        self.ui.image2.setText("原始印章的特徵")

        self.ui.model_image.clear()
        self.ui.model_image.setText(
            "與原始印章最相似的照片")

        self.ui.label.clear()
        self.ui.label.setText("原始印章的特徵")

        self.ui.analyz.setText("印章類別")
        self.ui.analyz_2.setText("相似度百分比")
        self.ui.analyz_3.setText("分析結論")

        self.ui.progressBar.setValue(0)

        self.model_path = None
        self.label_encoder_path = None
        self.gan_model_path = None
        self.generated_images_dir = None
        self.input_image_path = None


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    application.show()
    sys.exit(app.exec_())