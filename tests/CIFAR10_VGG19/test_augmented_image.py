"""
This file is used to test the augmented image class
"""
import unittest
from src.CIFAR10_VGG19.utils.augmented_images import AugmentedImages
from src.CIFAR10_VGG19.utils.load_image_from_folder import load_images_from_folder

class TestaugmentedImage(unittest.TestCase):
    """
    Test Augmented Image
    """
    def setUp(self):
        self.load_image_path = "dataset/Q1_image/Q1_1"
        self.imagelist = load_images_from_folder(self.load_image_path)
        self.augmented_image = AugmentedImages(self.imagelist)

    def test_augmented_images(self):
        """
        Test Augmented Images
        """
        self.assertIsNotNone(self.imagelist)
        self.assertEqual(len(self.imagelist), 9)
        self.assertIsNotNone(self.augmented_image.get_augmented_data())

if __name__ == "__main__":
    unittest.main()
