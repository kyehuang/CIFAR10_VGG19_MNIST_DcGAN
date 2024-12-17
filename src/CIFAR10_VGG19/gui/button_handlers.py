"""
This file contains the class button_handlers which contains the functions for the button 
click events.
"""
import matplotlib.pyplot as plt
import torch
import torchsummary
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from src.CIFAR10_VGG19.utils.load_image_from_folder import load_images_from_folder
from src.CIFAR10_VGG19.utils.augmented_images import AugmentedImages
from src.CIFAR10_VGG19.utils.inferce import inference
from src.CIFAR10_VGG19.utils.show_acc_loss import show_acc_loss
from src.CIFAR10_VGG19.utils.show_predicted_distribution import show_predicted_distribution
class ButtonHandlers:
    """
    This class contains the functions for the button click events.
    """
    def __init__(self, parent_widget):
        self.parent_widget = parent_widget

    def show_augmented_images(self):
        """
        Show Augmented Images
        """
        try:
            print("Show Augmented Images")
            imagelist = load_images_from_folder("dataset/Q1_image/Q1_1") # load images from folder
            augmented_images = AugmentedImages(imagelist)
            augmented_dict = augmented_images.get_augmented_data()      # get augmented images

            labels = list(augmented_dict.keys())
            images = list(augmented_dict.values())
            # Display augmented images
            plot_pil_images_grid(images, labels, grid_size=(3, 3), figsize=(8, 8))
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            path = "./models/vggModel.pth"
            model = torch.load(path)
            torchsummary.summary(model.to(device), (3, 32, 32))
            return True
        except ImportError as e:
            print("Error: ", e)
            return False

    def show_accuracy_loss(self):
        """
        Show Accuracy Loss
        """
        try:
            print("Show Accuracy Loss")
            image_acc_path = "src/CIFAR10_VGG19/result/Accuracy_20241217.png"
            image_loss_path = "src/CIFAR10_VGG19/result/Loss_20241217.png"
            show_acc_loss(image_acc_path=image_acc_path, image_loss_path=image_loss_path)
            return True
        except ImportError as e:
            print("Error: ", e)
            return False

    def do_inference(self, parrent ,image_path):
        """
        Inference
        """
        result = inference(image_path=image_path)

        pixmap = QPixmap(image_path)
        parrent.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        parrent.prediction_label.setText(f"Predicted = {result['most_likely']}")

        show_predicted_distribution(result['scores'])
        return result


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
    _, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()

    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis('on')

    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
