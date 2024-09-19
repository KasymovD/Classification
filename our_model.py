import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tensorflow.keras.models import load_model, Model
import joblib
import os
from PyQt5.QtGui import QImage
import io

def full_predict(image_path, model_path, label_encoder_path):
    model = load_model(model_path, compile=False)
    label_encoder = joblib.load(label_encoder_path)

    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

    img_size = (128, 128)
    pil_image = Image.open(image_path).convert('L').resize(img_size)
    img_array = np.array(pil_image)
    img_array_rgb = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
    img_array_normalized = img_array_rgb.astype('float32') / 255.0
    img_array_normalized = np.expand_dims(img_array_normalized, axis=0)

    predictions = model.predict(img_array_normalized)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

    label_parts = predicted_label.split('/')
    main_category = label_parts[0] if label_parts else predicted_label
    sub_categories = label_parts[1:] if len(label_parts) > 1 else []

    features_original = feature_extractor.predict(img_array_normalized)

    features_image = get_visual_features(pil_image)

    cluster_name = None
    similarity = 0
    most_similar_image = None
    features_similar = None
    features_similar_image = None

    if main_category == '公司':
        kmeans_model_path = 'Cluster_Dataset/clustered_images_company/kmeans_model.pkl'
        data_folder = 'Dataset/公司'
    elif main_category == '關防-整理好的':
        kmeans_model_path = 'Cluster_Dataset/clustered_images_government/kmeans_model.pkl'
        data_folder = 'Dataset/關防-整理好的'
    else:
        raise ValueError(f"未知類別: {main_category}")

    if not os.path.exists(kmeans_model_path):
        raise FileNotFoundError(f"未在路徑中找到 KMeans 模型: {kmeans_model_path}")

    kmeans = joblib.load(kmeans_model_path)

    cluster_id = kmeans.predict(features_original)[0]
    cluster_name = int(cluster_id)


    features_database = np.load(os.path.join(data_folder, 'features_database.npy'))
    filenames_database = joblib.load(os.path.join(data_folder, 'filenames_database.pkl'))

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(features_original, features_database)[0]

    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx] * 100

    if max_similarity >= 85:
        similarity = max_similarity
        most_similar_image_path = filenames_database[max_similarity_idx]
        features_similar = features_database[max_similarity_idx]

        similar_pil_image = Image.open(most_similar_image_path).convert('L').resize(img_size)
        features_similar_image = get_visual_features(similar_pil_image)
        most_similar_image = most_similar_image_path
    else:
        similarity = max_similarity
        most_similar_image = None
        features_similar = None
        features_similar_image = None

    return {
        'main_category': main_category,
        'sub_categories': sub_categories,
        'cluster_name': cluster_name,
        'similarity': similarity,
        'most_similar_image': most_similar_image,
        'features_original': features_original.flatten(),
        'features_similar': features_similar.flatten() if features_similar is not None else None,
        'features_image': features_image,
        'features_similar_image': features_similar_image
    }

def get_visual_features(pil_image):
    edge_image = pil_image.filter(ImageFilter.FIND_EDGES)

    byte_arr = io.BytesIO()
    edge_image.save(byte_arr, format='PNG')
    qimage = QImage()
    qimage.loadFromData(byte_arr.getvalue())

    return qimage