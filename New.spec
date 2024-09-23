# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Функция для рекурсивного сбора всех файлов из директории
def collect_all_files(source, destination):
    collected = []
    for root, dirs, files in os.walk(source):
        for file in files:
            # Получаем относительный путь
            rel_path = os.path.relpath(root, source)
            # Формируем путь назначения
            dest_path = os.path.join(destination, rel_path)
            # Полный путь к файлу
            src_file = os.path.join(root, file)
            collected.append((src_file, dest_path))
    return collected

# Список data файлов
datas = [
    ('main_ui.py', '.'),  # Добавляем main_ui.py
    ('our_model.py', '.'),  # Добавляем our_model.py
    ('model.h5', '.'),  # Модель для CNN
    ('label_encoder.pkl', '.'),  # Label Encoder файл
    ('training_history.pkl','.'),
    # Добавляем иконки
    ('icons', 'icons'),
    ('Cluster_Dataset/clustered_images_government/kmeans_model.pkl', 'Cluster_Dataset/clustered_images_government'),
    ('Dataset/公司/features_database.npy', '.'),
    ('Dataset/公司/filenames_database.pkl', '.'),
    ('Dataset/關防-整理好的/features_database.npy', '.'),
    ('Dataset/關防-整理好的/filenames_database.pkl', '.'),
]

# Добавляем файлы из Cluster_Dataset и Dataset
datas += collect_all_files('Cluster_Dataset/clustered_images_company', 'Cluster_Dataset/clustered_images_company')
datas += collect_all_files('Cluster_Dataset/clustered_images_government', 'Cluster_Dataset/clustered_images_government')
datas += collect_all_files('Dataset/公司', 'Dataset/公司')
datas += collect_all_files('Dataset/關防-整理好的', 'Dataset/關防-整理好的')

# Сбор скрытых импортов
hiddenimports = collect_submodules('scipy') + [
    'our_model',  # Ваш модуль
    'hachoir',  # Библиотека hachoir
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./hooks'],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='郵票分類器IPAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Если хотите скрыть консоль
    icon='favicon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='StampClassifier'
)
