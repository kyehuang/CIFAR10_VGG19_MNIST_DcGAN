"""
This file contains the class Augmented_images which is used to 
augment the images in the CIFAR10 dataset.
"""
import copy
from torchvision import transforms

class AugmentedImages:
    """
    This class is used to augment the images in the CIFAR10 
    dataset.
    """
    def __init__(self, images: list):
        self.images = copy.deepcopy(images)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
        ])
        self.augmented_data = []

        self.augment_data()

    def augment_data(self):
        """
        Augment the data
        """
        print("Augmenting Data")
        for i, img in enumerate(self.images):
            print(f"Augmenting {i + 1}")
            augmented_img = self.transform(img)
            self.augmented_data.append(augmented_img)

    def get_augmented_data(self):
        """
        Get the augmented data
        """
        return self.augmented_data
        