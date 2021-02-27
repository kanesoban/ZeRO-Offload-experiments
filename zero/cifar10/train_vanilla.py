import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from pynvml import *


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    nvmlInit()
    device_id = 0
    device_handle = nvmlDeviceGetHandleByIndex(device_id)

    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    info = nvmlDeviceGetMemoryInfo(device_handle)
    print('Total GPU memory for device before training {}: {}'.format(device, info.total))
    print('Free GPU memory for device before training {}: {}'.format(device, info.free))
    print('Used GPU memory for device before training {}: {}'.format(device, info.used))
    used_memory_0 = info.used

    net = models.resnet50()
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    used_memory = 0
    epochs = 1
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        info = nvmlDeviceGetMemoryInfo(device_handle)
        print('Total GPU memory for device {}: {}'.format(device, info.total))
        print('Free GPU memory for device {}: {}'.format(device, info.free))
        print('Used GPU memory for device {}: {}'.format(device, info.used))
        used_memory += info.used

    used_memory /= epochs
    print('Finished Training')
    print('Used memory by model (MB): {}'.format((used_memory - used_memory_0) / (1024 * 1024)))

    PATH = './.cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    test_model(classes, net, testloader)


def test_model(classes, net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    main()
