import matplotlib.pyplot as plt
from PIL import Image

def show_image(image_path: str) -> None:
    """Show image from path."""
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    show_image('src/DcGAN/result/Loss.png')
    show_image("src/DcGAN/result/real_fake.png")
