"""
This file contains the test cases for the load_image_from_folder.py file
"""
import unittest
from src.CIFAR10_VGG19.utils.load_image_from_folder import load_images_from_folder

class TestLoadImageFromFolder(unittest.TestCase):
    """
    Test Load Image From Folder
    """
    def setUp(self):
        self.load_image_path = "dataset/Q1_image/Q1_1"

    def test_load_image_from(self):
        """
        Test Load Image From Folder
        """
        images = load_images_from_folder(self.load_image_path)
        self.assertEqual(len(images), 9)
        self.assertIsNotNone(images, "Images are loaded successfully")


if __name__ == "__main__":
    unittest.main()
