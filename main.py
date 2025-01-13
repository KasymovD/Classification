import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QMessageBox
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from ui_main import Ui_MainWindow
import os
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QMessageBox, QInputDialog
import pickle
from utils import resource_path_1

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')
            img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"{image_path}: {e}")
        return None

def binarize_image(image_array, threshold=127):
    binarized = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return binarized

def calculate_similarity(image1_array, image2_array):
    if image1_array.shape != image2_array.shape:
        image2_array = cv2.resize(image2_array, (image1_array.shape[1], image1_array.shape[0]))
    total_pixels = image1_array.size
    matching_pixels = np.sum(image1_array == image2_array)
    similarity_percentage = (matching_pixels / total_pixels) * 100
    return similarity_percentage

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.showMaximized()
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        container_widget = QWidget()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        scroll_area.setWidget(self.centralWidget())
        self.setCentralWidget(scroll_area)
        self.ui.Start_button.clicked.connect(self.start_comparison)
        self.ui.Start_button_2.clicked.connect(self.clear_all)
        self.black_white_folder = 'black_white'
        self.original_folder = 'original'
        self.database_images = self.load_database_images()
        self.file_category_mapping = self.load_mapping('file_category_mapping.pkl')

        self.ui.input_image.setFixedSize(256, 256)
        # Повторите для всех QLabel, где отображаются изображения
        self.ui.input_image_2.setFixedSize(256, 256)
        self.ui.original_image.setFixedSize(256, 256)
        self.ui.original_image_2.setFixedSize(256, 256)
        self.ui.top1.setFixedSize(356, 256)
        self.ui.top1_black_white.setFixedSize(356, 256)
        self.ui.top2.setFixedSize(356, 256)
        self.ui.top2_black_white.setFixedSize(356, 256)
        self.ui.top3.setFixedSize(356, 256)
        self.ui.top3_black_white.setFixedSize(356, 256)



    def load_mapping(self, filename='file_category_mapping.pkl'):
        try:
            with open(filename, 'rb') as f:
                mapping = pickle.load(f)
            return mapping
        except FileNotFoundError:
            QMessageBox.warning(self, "錯誤", f"{filename}")
            return {}


    def load_database_images(self, size=(256, 256)):
        images = []
        folder = Path(self.black_white_folder)
        for image_file in folder.rglob('*'):
            if image_file.is_file() and image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                try:
                    img = self.load_image_with_pil(str(image_file), grayscale=True)
                    if img is not None:
                        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                        relative_path = image_file.relative_to(self.black_white_folder)
                        category = relative_path.parent
                        images.append((str(image_file), image_file.name, category, img_resized))
                except Exception as e:
                    print(f" {image_file}: {e}")
        return images

    def load_image_with_pil(self, image_path, grayscale=True):
        try:
            with Image.open(image_path) as img_pil:
                if grayscale:
                    img_pil = img_pil.convert('L')
                    img = np.array(img_pil)
                else:
                    img_pil = img_pil.convert('RGB')
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                return img
        except Exception as e:
            return None

    def compute_image_hash(self, image):
        import imagehash
        from PIL import Image

        pil_image = Image.fromarray(image)
        return imagehash.phash(pil_image)

    def start_comparison(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "選擇印章圖像",
            "",
            "Images (*.png *.jpg *.bmp)",
            options=options
        )
        if file_name:
            # Загружаем и масштабируем изображение
            pixmap = QPixmap(file_name).scaled(
                self.ui.input_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.input_image.setPixmap(pixmap)

            file_basename = os.path.basename(file_name).lower().strip()
            print(f"'{file_basename}'")
            categories = self.file_category_mapping.get(file_basename)
            if not categories:
                QMessageBox.information(self, "提示", f"未找到名稱為 {file_basename} 的相應分類，請手動選擇分類。")
                available_categories = self.get_available_categories()
                if not available_categories:
                    QMessageBox.warning(self, "錯誤", "未找到可用的類別")
                    return
                category, ok = QInputDialog.getItem(
                    self,
                    "選擇類別",
                    "請選擇類別：",
                    available_categories,
                    0,
                    False
                )
                if not ok:
                    QMessageBox.warning(self, "錯誤", "未選擇類別")
                    return
                categories = [category]
            else:
                categories = list(categories)

            if len(categories) > 1:
                category, ok = QInputDialog.getItem(
                    self,
                    "選擇類別",
                    f"文件 {file_basename}",
                    categories,
                    0,
                    False
                )
                if not ok:
                    QMessageBox.warning(self, "錯誤", "未選擇類別")
                    return
            else:
                category = categories[0]
            black_white_image_path = self.find_black_white_image(file_basename, category)
            if not black_white_image_path:
                QMessageBox.warning(self, "錯誤", f"未找到名稱為 {file_basename} 的相應黑白圖像 в категории {category}")
                return
            test_image = self.load_image_with_pil(black_white_image_path, grayscale=True)
            if test_image is None:
                return

            test_image = cv2.resize(test_image, (256, 256), interpolation=cv2.INTER_AREA)
            self.display_image_in_label(test_image, self.ui.input_image_2)
            selected_image_hash = self.compute_image_hash(test_image)

            category_images = []
            for img in self.database_images:
                image_path, name, cat, image = img
                if str(cat) == category:
                    current_image_hash = self.compute_image_hash(image)
                    if selected_image_hash != current_image_hash:
                        category_images.append(img)

            if len(category_images) < 2:
                self.ui.analyz.setText("該類別包含的圖像數量不足以進行比較❌")
                self.ui.original_image.setText("🚫")
                self.ui.original_image_2.setText("🚫")
                self.ui.top1.setText("🚫")
                self.ui.top1_black_white.setText("🚫")
                self.ui.top2.setText("🚫")
                self.ui.top2_black_white.setText("🚫")
                self.ui.top3.setText("🚫")
                self.ui.top3_black_white.setText("🚫")
                self.ui.analyz_2.setText("分析結論\n字體差異: 同字體 (不同字體)\n字樣差異: 同字樣 (不同字樣_\n字距差異: 字距相同(字距不同)\n行距差異性: 行距相同(行距不同)\n")
                return
            else:
                self.ui.analyz.setText("")
                self.ui.top1.clear()
                self.ui.top1_black_white.clear()
                self.ui.top2.clear()
                self.ui.top2_black_white.clear()
                self.ui.top3.clear()
                self.ui.top3_black_white.clear()

            similarities = []
            for image_path, name, cat, image in category_images:
                test_image_bin = binarize_image(test_image)
                image_bin = binarize_image(image)
                similarity = calculate_similarity(test_image_bin, image_bin)

                similarities.append((similarity, image_path, name, image))
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_similarities = similarities[:3]

            for idx, (similarity, image_path, name, image) in enumerate(top_similarities):
                original_image_path = self.get_original_image_path(image_path)
                if original_image_path and os.path.exists(original_image_path):
                    original_pixmap = QPixmap(original_image_path)
                    original_pixmap_resized = original_pixmap.scaled(256, 256, Qt.KeepAspectRatio)
                else:
                    original_pixmap_resized = QPixmap(256, 256)
                    original_pixmap_resized.fill(Qt.gray)

                highlighted_image = self.highlight_differences(image, test_image)
                highlighted_image_resized = cv2.resize(highlighted_image, (256, 256), interpolation=cv2.INTER_AREA)

                if idx == 0:
                    self.ui.original_image.setPixmap(original_pixmap_resized)
                    self.display_image_in_label(highlighted_image_resized, self.ui.original_image_2)
                    self.ui.top1.setPixmap(original_pixmap_resized)
                    self.display_image_in_label(highlighted_image_resized, self.ui.top1_black_white)
                elif idx == 1:
                    # Top 2
                    self.ui.top2.setPixmap(original_pixmap_resized)
                    self.display_image_in_label(highlighted_image_resized, self.ui.top2_black_white)
                elif idx == 2:
                    # Top 3
                    self.ui.top3.setPixmap(original_pixmap_resized)
                    self.display_image_in_label(highlighted_image_resized, self.ui.top3_black_white)

            category_name = category.replace(os.sep, ' -> ')
            analysis_text = f"類別名稱: {category_name}\n\n"
            for idx, (similarity, image_path, name, image) in enumerate(top_similarities):
                similarity_percent = round(similarity, 2)
                analysis_text += f"Top {idx + 1}: {name} 相似度: {similarity_percent}%\n"
            self.ui.analyz.setText(analysis_text)

            font_difference = self.get_font_difference(test_image, top_similarities[0][3])
            spacing_difference = self.get_spacing_difference(test_image, top_similarities[0][3])

            different_pixels, total_pixels, difference_metric = self.calculate_difference_metrics(test_image,
                                                                                                  top_similarities[0][
                                                                                                      3])

            analysis_text_2 = (
                f"分析結論\n"
                f"字體差異: {font_difference}\n"
                f"字樣差異: {font_difference}\n"
                f"字距差異: {spacing_difference}\n"
                f"行距差異性: {spacing_difference}\n"
                # f"總像素數: {total_pixels}\n"
                # f"相同像素數: {int(total_pixels - different_pixels)}\n"
                # f"不同像素數: {different_pixels}\n"
                # f"差異百分比: {round(difference_metric, 2)}%\n"
                f"相似度得分: {round(top_similarities[0][0], 2)}%\n"
            )
            self.ui.analyz_2.setText(analysis_text_2)


    def get_category_from_path(self, file_path):
        try:
            relative_path = os.path.relpath(file_path, self.original_folder)
            category = os.path.dirname(relative_path)
            return category
        except Exception as e:
            return None

    def find_black_white_image(self, file_basename, category=None):
        if category:
            potential_path = os.path.join(self.black_white_folder, category, file_basename)
            if os.path.exists(potential_path):
                return potential_path
        return None

    def get_corresponding_black_white_image_path(self, original_image_path):
        relative_path = os.path.relpath(original_image_path, self.original_folder)
        black_white_image_path = os.path.join(self.black_white_folder, relative_path)
        return black_white_image_path

    def get_original_image_path(self, black_white_image_path):
        relative_path = os.path.relpath(black_white_image_path, self.black_white_folder)

        original_image_path = os.path.join(self.original_folder, relative_path)
        if not os.path.exists(original_image_path):
            print(f"{original_image_path}")
            return None
        return original_image_path

    def display_image_in_label(self, image, label):
        if len(image.shape) == 2:
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def calculate_similarity(self, img1, img2):
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        match_pixels = np.count_nonzero(img1 == img2)
        total_pixels = img1.size
        similarity = match_pixels / total_pixels
        return similarity

    def highlight_differences(self, img1, img2):
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        _, img1_thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        _, img2_thresh = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(img1_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            border_mask = np.zeros_like(img1_thresh)
            cv2.drawContours(border_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
            inner_mask = cv2.bitwise_not(border_mask)
        else:
            inner_mask = np.ones_like(img1_thresh) * 255
            border_mask = np.zeros_like(img1_thresh)

        border_diff = cv2.bitwise_and(cv2.absdiff(img1_thresh, img2_thresh), border_mask)
        inner_diff = cv2.bitwise_and(cv2.absdiff(img1_thresh, img2_thresh), inner_mask)
        img_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_color[border_diff > 50] = [0, 0, 255]  # BGR: красный
        img_color[inner_diff > 50] = [0, 255, 0]  # BGR: зеленый

        return img_color

    def calculate_difference_metrics(self, img1, img2):
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        diff = cv2.absdiff(img1, img2)
        different_pixels = np.count_nonzero(diff)
        total_pixels = diff.size
        difference_metric = (different_pixels / total_pixels) * 100
        return different_pixels, total_pixels, difference_metric

    def get_font_difference(self, img1, img2):
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        _, img1_thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        _, img2_thresh = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

        contours1, _ = cv2.findContours(img1_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(img2_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moments1 = cv2.moments(img1_thresh)
        huMoments1 = cv2.HuMoments(moments1).flatten()
        moments2 = cv2.moments(img2_thresh)
        huMoments2 = cv2.HuMoments(moments2).flatten()
        difference = np.sum(np.abs(huMoments1 - huMoments2))
        difference = np.log10(difference + 1)
        if difference < 0.5:
            return "字體相同"
        elif difference < 1.0:
            return "字體相似"
        else:
            return "字體不同"

    def get_spacing_difference(self, img1, img2):
        projection1 = np.sum(img1 == 0, axis=0)
        projection2 = np.sum(img2 == 0, axis=0)
        projection1 = projection1 / np.max(projection1)
        projection2 = projection2 / np.max(projection2)

        diff = np.abs(projection1 - projection2)
        mse = np.mean(diff**2)

        if mse < 0.01:
            return "字距相同"
        elif mse < 0.05:
            return "字距相似"
        else:
            return "字距不同"

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            reply = QMessageBox(self)
            reply.setWindowTitle("確認退出")
            reply.setText("您確定要關閉應用程式嗎？")
            reply.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            yes_button = reply.button(QMessageBox.Yes)
            yes_button.setText("是")
            no_button = reply.button(QMessageBox.No)
            no_button.setText("否")

            reply.exec()

            if reply.clickedButton() == yes_button:
                self.close()

    def clear_all(self):
        self.ui.input_image.setText("欲辨識印章")
        self.ui.input_image_2.setText("欲辨識印章特徵")
        self.ui.original_image.setText("最相似印章")
        self.ui.original_image_2.setText("最相似印章特徵")
        self.ui.top1.clear()
        self.ui.top1_black_white.clear()
        self.ui.top2.clear()
        self.ui.top2_black_white.clear()
        self.ui.top3.clear()
        self.ui.top3_black_white.clear()

        self.ui.analyz.setText("類別名稱")
        self.ui.analyz_2.setText("分析結論\n字體差異: 同字體 (不同字體)\n字樣差異: 同字樣 (不同字樣_\n字距差異: 字距相同(字距不同)\n行距差異性: 行距相同(行距不同)\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())