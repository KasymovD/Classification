# utils.py

import sys
import os

def resource_path(relative_path):
    """Получает абсолютный путь к ресурсному файлу, работает как в режиме разработки, так и в собранном приложении"""
    try:
        # PyInstaller создает временную папку и сохраняет путь в _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
