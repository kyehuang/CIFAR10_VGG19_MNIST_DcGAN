"""
This file contains the class button_handlers which contains the functions for the button 
click events.
"""
import matplotlib.pyplot as plt
import torch
import torchsummary

from src.DcGAN.utils.augmented_images import augmented_images
from src.DcGAN.DcGAN_model.build_dcGAN import build_generator, build_discriminator
from src.DcGAN.utils.show_image import show_image


class ButtonHandlers:
    """
    This class contains the functions for the button click events.
    """
    def __init__(self, parent_widget):
        self.parent_widget = parent_widget

    def show_training_images(self):
        """
        Show Training Images
        """
        try:
            print("Show Training Images")
            augmented_images()
            return True
        except FileNotFoundError as e:
            print("Error: ", e)
            return False
        except ImportError as e:
            print("Error: ", e)
            return False

    def show_model_structure(self):
        """
        Show Model Structure
        """
        try:
            print("Show Model Structure")
            generator = build_generator(1)
            discriminator = build_discriminator(1)
            print(generator)
            print(discriminator)
            return True
        except ImportError as e:
            print("Error: ", e)
            return False

    def show_training_loss(self):
        """
        Show Training Loss
        """
        try:
            print("Show Training Loss")
            show_image("src/DcGAN/result/Loss.png")
            return True
        except ImportError as e:
            print("Error: ", e)
            return False

    def inference(self):
        """
        Inference
        """
        try:
            print("Inference")
            show_image("src/DcGAN/result/real_fake.png")
            return True
        except ImportError as e:
            print("Error: ", e)
            return False

def plot_pil_images_grid(images, labels, grid_size=(3, 3), figsize=(8, 8)):
    """
    Plot a grid of PIL images.

    Args:
    images: List of PIL images.
    labels: List of image labels.
    grid_size: Tuple of grid size (rows, columns).
    figsize: Tuple of figure size (width, height).

    Returns:
    None
    """
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()  

    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis('on')

    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
