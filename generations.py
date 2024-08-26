import os
import numpy as np
from PIL import Image, ImageEnhance


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    return image


def augment_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(np.random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(np.random.uniform(0.8, 1.2))

    image_array = np.array(image)
    noise = np.random.normal(0, 3, image_array.shape).astype(np.uint8)
    image = Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))

    image = image.rotate(np.random.uniform(0, 360))

    return image


def augment_images_in_folder(folder_path, augment_count=30):
    images = []
    labels = []

    folder_name = os.path.basename(folder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = load_image(img_path)

            base_filename = os.path.splitext(filename)[0]

            for i in range(augment_count):
                augmented_image = augment_image(image)
                augmented_filename = f"{base_filename}_augmentation_{i + 1}.png"
                augmented_image.save(os.path.join(folder_path, augmented_filename))

                images.append(np.array(augmented_image))
                labels.append(folder_name)

    print(f"Аугментация изображений в папке {folder_path} завершена.")

    return np.array(images), np.array(labels)


base_folder = 'Processed_Dataset'
all_images = []
all_labels = []

for root, dirs, files in os.walk(base_folder):
    for subfolder in dirs:
        folder_path = os.path.join(root, subfolder)
        if os.path.isdir(folder_path):
            images, labels = augment_images_in_folder(folder_path, augment_count=30)
            if images.size > 0:
                all_images.append(images)
                all_labels.extend(labels)

if all_images:
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)

print("Аугментация всех изображений завершена.")