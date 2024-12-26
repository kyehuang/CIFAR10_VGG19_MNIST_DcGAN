import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
import torchvision

def imshow(img, title, ax):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(title)
    ax.axis('off')

def augmented_images():
    # 定义数据变换
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=60),
    transforms.ToTensor(),
    ])


    dataroot = "dataset/Q2_image"
    dataset = datasets.ImageFolder(root=dataroot, transform=transform)


    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    dataiter = iter(dataloader)
    images, _ = next(dataiter)


    to_pil = transforms.ToPILImage()
    augment_images = torch.stack([augment_transform(to_pil(img)) for img in images])


    _, axes = plt.subplots(1, 2, figsize=(10, 5))  # 创建1行2列的子图

    imshow(torchvision.utils.make_grid(images), "Training Dataset (Original)",axes[0])   # 显示原始图像
    imshow(torchvision.utils.make_grid(augment_images), "Training Dataset (Augmented)",axes[1])  # 显示增强图像

    plt.tight_layout()
    plt.show()
