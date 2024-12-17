"""
Time: 2024/12/17
Author: Kye Huang
"""
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import time


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    transform = transforms.Compose(
        [v2.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         v2.RandomRotation(30),
         transforms.ToTensor(),
         transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    BATCH_SIZE = 100

    trainSet = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True,
                                            download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2, pin_memory=True)

    testSet = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=False,
                                           download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = torch.load("./models/vggModel.pth", map_location=device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], gamma=0.2)
    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []
    startTime = time.time()
    current = datetime.now().strftime("%Y%m%d-%H%M%S")
    EPOCHS = 120
    pbar = tqdm(range(EPOCHS), desc="Epoch")
    BESTACC = 0.0
    for epoch in pbar:
        model.train()
        RUNNING_LOSS = 0.0
        CORRECT_TRAIN = 0
        TOTAL_TRAIN = 0
        for inputs, labels in tqdm(trainLoader, unit="images",
                                   unit_scale=trainLoader.batch_size, leave=False, desc="Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            RUNNING_LOSS += loss.item()
            _, predicted = outputs.max(1)
            TOTAL_TRAIN += labels.size(0)
            CORRECT_TRAIN += predicted.eq(labels).sum().item()

        trainLoss.append(RUNNING_LOSS / len(trainLoader))
        trainAcc.append(100*CORRECT_TRAIN / TOTAL_TRAIN)

        model.eval()
        RUNNING_TEST_LOSS = 0.0
        CORRECT_TEST = 0
        TOTAL_TEST = 0
        with torch.no_grad():
            for inputs, labels in tqdm(testLoader, unit="images",
                                       unit_scale=testLoader.batch_size, leave=False, desc="Test"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                RUNNING_TEST_LOSS += loss.item()
                _, predicted = outputs.max(1)
                TOTAL_TEST += labels.size(0)
                CORRECT_TEST += predicted.eq(labels).sum().item()

        valLoss.append(RUNNING_TEST_LOSS / len(testLoader))
        valAcc.append(100*CORRECT_TEST / TOTAL_TEST)
        pbar.set_postfix({
            "TrainLoss": trainLoss[-1],
            "TrainAcc": trainAcc[-1],
            "ValidLoss": valLoss[-1],
            "ValidAcc": valAcc[-1]
        })
        lr_scheduler.step()
        if valAcc[-1] > BESTACC:
            torch.save(model, f"./models/trained_model_{current}")

    endTime = time.time() - startTime
    print(f'\n The Training Took {endTime} seconds')

    f = plt.figure(1)
    plt.plot(range(1, EPOCHS+1), trainLoss, color='blue', label='TrainLoss')
    plt.plot(range(1, EPOCHS+1), valLoss, color='orange', label='valLoss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.show()
    plt.savefig(f"./plots/Loss_{current}.png")

    g = plt.figure(2)
    plt.plot(range(1, EPOCHS+1), trainAcc, color='blue', label='TrainAcc')
    plt.plot(range(1, EPOCHS+1), valAcc, color='orange', label='valAcc')
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig(f"./plots/Accuracy_{current}.png")
