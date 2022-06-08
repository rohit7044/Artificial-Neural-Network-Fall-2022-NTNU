# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import pickleStore, readData
from preprocessing.preprocessing import preprocess, transform_dataset, train_test_split
from dataset.dataset import Dataset
from model.model import LSTMPredictor
from trainer.supervised import trainer, tester

from torch.utils.data.dataloader import default_collate

import os
import argparse


"""
# Project#1 Keras Tutorial: Stock prediction

2022/3/2 Neural Network

For your references:

*   [Pytorch official website](https://pytorch.org/)

*   [Google Colab official tutorial](https://colab.research.google.com/notebooks/welcome.ipynb?hl=zh-tw#scrollTo=gJr_9dXGpJ05)

*   [Using outer files in Google colab](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=BaCkyg5CV5jF)

"""


if __name__:


    ## Parser initializing
    parser = argparse.ArgumentParser(description='Train prediction model')
    parser.add_argument('--ngpu', default=1, type=int, required=False)
    args   = parser.parse_args()


    ## Device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    # device = torch.device("cuda:1" if args.ngpu > 0 else "cpu")
    # device = torch.device("cpu")


    ## Data
    data = readData("./data/1795_history.csv")
    print('Num of samples:', len(data))


    ## Preprocess
    prices = preprocess(data)
    # Divide trainset and test set
    train, test = train_test_split(prices, 0.8)
    # Set the N(look_back)=5
    look_back = 5
    trainX, trainY = transform_dataset(train, look_back)
    testX, testY = transform_dataset(test, look_back)
    # print(trainX)
    # Get dataset
    trainset = Dataset(trainX, trainY, device)
    testset  = Dataset(testX, testY, device)
    # Get dataloader
    batch_size = 200
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers should set 1 if put data on CUDA
    testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


    ## Model
    net = LSTMPredictor(look_back)
    net.to(device)


    ## Loss function
    criterion = nn.MSELoss(reduction = "mean")


    ## Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.1)


    ## Training
    checkpoint = "./checkpoint/save.pt"
    if not os.path.isfile(checkpoint):
        trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=500, path=checkpoint)
    else:
        net.load_state_dict(torch.load(checkpoint, map_location='cuda'))


    ## Test the model
    test = tester(net, criterion, testloader)
    # Show the difference between predict and groundtruth (loss)
    print('Test Result: ', test)


    ## Predict
    predict = net.predict(torch.tensor([[126, 124, 124, 122.5, 121]], dtype=torch.float32).to(device))
    print('Predict Result', predict)