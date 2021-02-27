import os
import time
import argparse

from tqdm import tqdm
import torch
import torchtext
from torchtext.datasets import text_classification
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import deepspeed


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngrams', default=2, type=int)
    parser.add_argument('--data_dir', default='./.data')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--embed_dim', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_, optimizer, model, device, criterion, scheduler, batch_size):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def train_func_deepspeed(sub_train_, optimizer, model, device, criterion, scheduler, batch_size):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        #optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        model.backward(loss)
        model.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(data_, device, model, criterion, batch_size):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


def main(args):
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=args.data_dir, ngrams=args.ngrams,
                                                                          vocab=None)
    vocab_size = len(train_dataset.get_vocab())
    device = torch.device("cuda")
    num_class = len(train_dataset.get_labels())

    model = TextSentiment(vocab_size, args.embed_dim, num_class).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.deepspeed:
        # train_deepspeed(args, model, train_dataset, criterion)
        optimizer = None
        scheduler = None
        train_deepspeed(args, criterion, device, model, optimizer, scheduler, test_dataset, train_dataset)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        train(args, criterion, device, model, optimizer, scheduler, test_dataset, train_dataset)


'''
def train_deepspeed(args, model, train_dataset, criterion):
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters,
                                                                   training_data=train_dataset)

    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].to(model_engine.device)
        labels = data[1].to(model_engine.device)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
'''


def train(args, criterion, device, model, optimizer, scheduler, test_dataset, train_dataset):
    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_, optimizer, model, device, criterion, scheduler, args.batch_size)
        valid_loss, valid_acc = test(sub_valid_, device, model, criterion, args.batch_size)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
    print('Checking the results of test dataset...')
    test_loss, test_acc = test(test_dataset, device, model, criterion, args.batch_size)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


def train_deepspeed(args, criterion, device, model, optimizer, scheduler, test_dataset, train_dataset):
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters,
                                                         training_data=train_dataset)

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    model_engine.train()
    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        train_loss, train_acc = train_func_deepspeed(sub_train_, optimizer, model_engine, device, criterion, scheduler,
                                                     args.batch_size)
        # valid_loss, valid_acc = test(sub_valid_, device, model, criterion, args.batch_size)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
    # print('Checking the results of test dataset...')
    # test_loss, test_acc = test(test_dataset, device, model, criterion, args.batch_size)
    # print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


if __name__ == "__main__":
    args = parse_args()
    main(args)
