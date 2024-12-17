"""
Time 2024/12/17
Author: Kye Huang
"""
import torch
import torchvision.transforms as transforms
from PIL import Image

def inference(model_path: str = "./models/VGG.pth"
            , image_path: str = "dataset/Q1_image/Q1_1/automobile.png") -> dict:
    """
    Inferce the image using the model

    Args:
    model_path (str): the path to the model
    image_path (str): the path to the image

    Returns:
    dict: the scores of each class and the most likely class
    """
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    model = torch.load(model_path, map_location=device).to(device)

    # set model to evaluation mode
    model.eval()

    # # define transform, same as CIFAR-10 test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    classes = ('plane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # read image
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return {"error": "File not found"}
    image = transform_test(image)
    image = image.unsqueeze(0)

    # send image to device
    image = image.to(device)

    # inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    # the score of each class
    scores = {classes[i]: prob.item() for i, prob in enumerate(probabilities.squeeze())}

    # the most likely class
    most_likely = classes[predicted.item()]

    return {"scores": scores, "most_likely": most_likely}


if __name__ == "__main__":
    result = inference()
    print(result)
