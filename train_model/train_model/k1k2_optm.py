import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt 
import time
import csv
import os
from scipy.stats import qmc

import sys
sys.path.append("../..") 
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
        self.input_layer3 = nn.Linear(12, 12)
        self.output_layer = nn.Linear(12, 2)
        self.activate1 = nn.Sigmoid()
        self.activate2 = nn.Sigmoid()
        self.activate3 = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer1(x)
        x = self.activate1(x)
        x = self.input_layer2(x)
        x = self.activate2(x)
        x = self.input_layer3(x)
        x = self.activate3(x)
        x = self.output_layer(x)
        outputs = x
        return outputs


# model train
def train_loop(dataloader, model, loss_fn, optimizer, t, train_loss_list):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    epochs_loss = 0
    loss = 0
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred.squeeze(), y)
        epochs_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    epochs_loss /= num_batchs
    print(f"Train Error: \n Avg loss: {epochs_loss:>8f}")
    train_loss_list.append(epochs_loss)


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
    epochs = 2500
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    loss_fn = nn.HuberLoss()
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
    torch.save(model, 'temp/model-optm-' + time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime(time.time()))  + '.pth')

    interval = 100
    # Calculate averaged data
    x = range(0, epochs)
    y1 = train_loss_list
    y2 = val_loss_list

    x_plot = list(range(interval, epochs + 1, interval))  # Epoch numbers for averages
    y1_plot = []
    y2_plot = []

    for i in range(0, epochs, interval):
        end_index = min(i + interval, epochs)
        y1_plot.append(np.mean(train_loss_list[i:end_index]))
        y2_plot.append(np.mean(val_loss_list[i:end_index]))


    # Create directory if it doesn't exist
    output_dir = './analyse_train_process'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save averaged train process data to CSV
    with open(os.path.join(output_dir, 'loss_data_'+ time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime(time.time())) + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Average Train Loss', 'Average Validation Loss'])
        for i in range(len(x_plot)):
            writer.writerow([x_plot[i], y1_plot[i], y2_plot[i]])

    # Loss graph
    plt.plot(x_plot, y1_plot, '.-', label='Train loss')
    plt.plot(x_plot, y2_plot, '.-', label='Validation loss')
    plt.title('Train and validation loss vs. epochs (Averaged every {} epochs)'.format(interval))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create 'temp' directory if it doesn't exist
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    plt.savefig(os.path.join(temp_dir, 'history.png'), format='png')
    plt.show()

    print("Averaged loss data saved to {}/loss_data.csv".format(output_dir))
    print("Loss plot saved to {}/history.png".format(temp_dir))

    print("Done!")


if __name__ == '__main__':
    train_path = '../train_data/high_kpa_data.csv'
    val_path = '../train_data/test.csv'
    train_model(train_path, val_path)