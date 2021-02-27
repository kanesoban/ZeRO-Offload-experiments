import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import torch.nn as nn
from pynvml import *
import torchvision.models as models

from zero.fp16.fp16 import FP16_Module


def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')

    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


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

    nvmlInit()
    device_id = 0
    device_handle = nvmlDeviceGetHandleByIndex(device_id)

    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    info = nvmlDeviceGetMemoryInfo(device_handle)
    print('Total GPU memory for device before training {}: {}'.format(device, info.total))
    print('Free GPU memory for device before training {}: {}'.format(device, info.free))
    print('Used GPU memory for device before training {}: {}'.format(device, info.used))
    used_memory_0 = info.used

    net = FP16_Module(models.resnet50())
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    args = add_argument()

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset)

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    used_memory = 0
    epochs = 1
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
                model_engine.local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

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

    test_model(classes, model_engine, net, testloader)


def test_model(classes, model_engine, net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(model_engine.local_rank))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(
                model_engine.local_rank)).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(model_engine.local_rank))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(model_engine.local_rank)).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    main()


