"""
This module provides a function to load images from a specified folder.
"""
import os
from PIL import Image


def load_images_from_folder(folder, num_images=9):
    """
    從指定資料夾中加載圖像。
    
    :param folder: 資料夾路徑
    :param num_images: 加載的圖像數量（默認為 9）
    :return: PIL Image 對象的列表
    """
    images = []
    try:
        files = os.listdir(folder)
        image_files = [f for f in files if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]

        for i, file in enumerate(image_files[:num_images]):
            image_path = os.path.join(folder, file)
            image = Image.open(image_path)
            images.append(image)
            print(f"Loaded {i + 1}: {image_path}")

    except FileNotFoundError as e:
        print(f"Error loading images: {e}")

    return images
