import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import FineGrainedModel
from train import train, test

def imshow(img):
    img = img / 2 + 0.5  # 非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # # 查看训练集和测试集的大小
    # train_size = len(trainloader.dataset)
    # test_size = len(testloader.dataset)

    # print(f'Train dataset size: {train_size}')
    # print(f'Test dataset size: {test_size}')

    # # 获取一些随机的训练图像
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)

    # # 显示图像
    # imshow(torchvision.utils.make_grid(images))

    # # 打印标签
    # print(' '.join('%5s' % trainset.classes[labels[j]] for j in range(4)))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FineGrainedModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, device, trainloader, criterion, optimizer, epoch)
        test(model, device, testloader, criterion)

    print('Finished Training')

if __name__ == '__main__':
    main()
