import os
from PIL import Image
import numpy as np


def load_and_preprocess_images(folder, output_folder, img_size=(128, 128)):
    images = []
    labels = []
    filenames = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder):
        for filename in files:
            img_path = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, folder)
            output_subfolder = os.path.join(output_folder, rel_dir)

            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            try:
                pil_image = Image.open(img_path).convert('L')
                pil_image = pil_image.resize(img_size)
                img = np.array(pil_image)
                images.append(img)

                labels.append(rel_dir)
                filenames.append(filename)

                output_path = os.path.join(output_subfolder, filename)
                pil_image.save(output_path)
            except Exception as e:
                print(f"Warning: Could not open or process the image {img_path}. Error: {e}")

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    return images, np.array(labels), filenames


data_folder = r'Dataset/公司'
output_folder = r'Processed_Dataset/公司'
images, labels, filenames = load_and_preprocess_images(data_folder, output_folder)

data_folder_2 = r'Dataset/關防-整理好的'
output_folder_2 = r'Processed_Dataset/關防-整理好的'
images_2, labels_2, filenames_2 = load_and_preprocess_images(data_folder_2, output_folder_2)

if images.ndim == images_2.ndim:
    images = np.concatenate((images, images_2), axis=0)
    labels = np.concatenate((labels, labels_2), axis=0)
else:
    print(f"Dimension mismatch: images has {images.ndim} dimensions, images_2 has {images_2.ndim} dimensions.")
