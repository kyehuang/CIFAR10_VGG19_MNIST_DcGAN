"""
This module provides a function to load images from a specified folder.
"""
import os
from PIL import Image


def load_images_from_folder(folder, num_images=9):
    """
    Load images from a folder
    :param folder: folder path
    :param num_images: number of images to load
    :return: list of images
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
