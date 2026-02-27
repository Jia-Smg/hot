import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt 
import time
from scipy.stats import qmc

import sys
sys.path.append("..") 
import utils

class Mydata(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path).values
    
    def __getitem__(self,idx):
        features = torch.tensor(self.df[idx, 1:13], dtype=torch.float32)
        kap = torch.tensor(self.df[idx, 13:15], dtype=torch.float32)
        return features, kap

    def __len__(self):
        data_lens = self.df.shape[0]
        return data_lens


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_layer1 = nn.Linear(12, 12)
        self.input_layer2 = nn.Linear(12, 12)
        self.output_layer = nn.Linear(12, 2)
        self.activate1 = nn.Sigmoid()
        self.activate2 = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer1(x)
        x = self.activate1(x)
        x = self.input_layer2(x)
        x = self.activate2(x)
        x = self.output_layer(x)
        outputs = x
        return outputs


# model train
def train_loop(dataloader, model, loss_fn, optimizer, t, train_loss_list):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    loss = 0
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss_list.append(loss.item())


# model test
def test_loop(dataloader, model, loss_fn, val_loss_list):
    num_batchs = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred.squeeze(), y).item()
    
    test_loss /= num_batchs
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    val_loss_list.append(test_loss)


def train_model(train_path, val_path):
    # data
    dataset1 = Mydata(train_path)
    train_data = DataLoader(dataset1, batch_size = 10, shuffle = True, drop_last = True)

    dataset2 = Mydata(val_path)
    val_data = DataLoader(dataset2, batch_size = 10, shuffle = True, drop_last=True)

    # model
    model = Net()

    # model setting
    learning_rate = 1e-3
    epochs = 1500
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loss graph
    train_loss_list = []
    val_loss_list = []

    # training
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train_loop(train_data, model, loss_fn, optimizer, t, train_loss_list)
        val_loss = test_loop(val_data, model, loss_fn, val_loss_list)
    
    # save model 
    torch.save(model, 'temp/model-' + time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime(time.time()))  + '.pth')

    # loss graph
    x = range(0, epochs)
    y1 = train_loss_list
    y2 = val_loss_list
    plt.plot(x, y1, '.-', label='Train loss')
    plt.plot(x, y2, '.-', label='Validation loss')
    plt.title('Train and validation loss vs. epoches' )
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('temp/history.png', format='png')

    print("Done!")


if __name__ == '__main__':
    train_path = 'train_data/high_kpa_data.csv'
    val_path = 'train_data/test.csv'
    train_model(train_path, val_path)