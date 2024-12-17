"""
Time: 2024/12/17 01:46
Author: Kye Huang
"""
import matplotlib.pyplot as plt
from PIL import Image


def show_acc_loss(
        image_acc_path: str = 'src/CIFAR10_VGG19/result/Accuracy_20241217.png',
        image_loss_path: str = 'src/CIFAR10_VGG19/result/Loss_20241217.png'):
    """
    show the accuracy and loss image

    Args:
    image_acc_path (str): the path to the accuracy image
    image_loss_path (str): the path to the loss image

    Returns:
    None
    """

    # Load the images (replace 'image1.jpg' and 'image2.jpg' with your image paths)
    try:
        image1 = Image.open(image_acc_path)
        image2 = Image.open(image_loss_path)
    except FileNotFoundError as e:
        print("Error: ", e)
        return
    image1 = Image.open(image_acc_path)
    image2 = Image.open(image_loss_path)

    # Create a figure to display the images
    plt.figure(figsize=(10, 5))

    # Display the first image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    # plt.title('Image 1')
    plt.axis('off')  # Turn off axis labels

    # Display the second image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    # plt.title('Image 2')
    plt.axis('off')  # Turn off axis labels

    # Show the images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_acc_loss()
