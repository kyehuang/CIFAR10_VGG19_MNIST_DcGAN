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
    :return: dictionary of images {label: image}
    """
    images_dict = {}
    try:
        files = os.listdir(folder)
        image_files = [f for f in files if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]

        for i, file in enumerate(image_files[:num_images]):
            image_path = os.path.join(folder, file)
            image = Image.open(image_path)

            # use filename as label
            label = os.path.splitext(file)[0]

            # add image to dictionary
            images_dict[label] = image
            print(f"Loaded {i + 1}: {image_path}")

        print(f"Loaded {len(images_dict)} images")
        print(f"Images: {images_dict.keys()}")

    except FileNotFoundError as e:
        print(f"Error loading images: {e}")

    return images_dict
