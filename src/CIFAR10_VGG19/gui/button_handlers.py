"""
This file contains the class button_handlers which contains the functions for the button 
click events.
"""
import matplotlib.pyplot as plt

from src.CIFAR10_VGG19.utils.load_image_from_folder import load_images_from_folder
from src.CIFAR10_VGG19.utils.augmented_images import AugmentedImages

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
            return True
        except ImportError as e:
            print("Error: ", e)
            return False

def plot_pil_images_grid(images, labels, grid_size=(3, 3), figsize=(8, 8)):
    """
    在網格中顯示 PIL 圖像及其標籤
    :param images: PIL.Image.Image 對象的列表
    :param labels: 圖像對應的標籤列表
    :param grid_size: (行數, 列數)，默認為 (3, 3)
    :param figsize: 圖形大小
    """
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()  # 展平成一維數組

    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image)  # 顯示 PIL 圖像
        axes[i].set_title(label, fontsize=10)
        axes[i].axis('off')  # 關閉坐標軸

    # 隱藏多餘的子圖
    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
