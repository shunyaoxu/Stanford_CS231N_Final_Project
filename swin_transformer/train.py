#%% Setup
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
from swin_transformer import *
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import sampler
from sklearn.model_selection import train_test_split
from os import listdir
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import argparse

parser = argparse.ArgumentParser(description='SWIN')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.000001,
                    help='learning rate decay')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% Use GPU
USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print('using device:', device)

#%% Load Data

#Split data into training, validation and test with proportion 8:1:1
PATH_OF_DATA = '/home/users/shunyaox/dataset/data/'
data_transforms = T.Compose([
                    #T.CenterCrop(1120),
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    ])
image_datasets = dset.ImageFolder(root=PATH_OF_DATA, transform=data_transforms)
total_size = len(image_datasets)
training_size = int(total_size * 0.8)
validation_size = int(total_size * 0.1)
test_size = total_size - training_size - validation_size
train, val, test = torch.utils.data.random_split(image_datasets, [training_size, validation_size, test_size])

#Load data with dataloaders, define batch_size here
trainLoader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
valLoader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)
print("Number of Training Data: ", len(train))
print("Number of Validation Data: ", len(val))
print("Number of Test Data: ", len(test))

#%% Check Accuracy
history = {}
history["val_acc"] = []
history["val_0_acc"] = []
history["val_1_acc"] = []
history["val_2_acc"] = []
history["val_3_acc"] = []
history["val_4_acc"] = []
history["train_loss"] = []
history["train_acc"] = []
history["train_0_acc"] = []
history["train_1_acc"] = []
history["train_2_acc"] = []
history["train_3_acc"] = []
history["train_4_acc"] = []
def check_accuracy(loader, model, train=True, val=False, test=False):
    if train:
        print('###Checking accuracy on train set###')
    elif val:
        print('###Checking accuracy on val set###')
    else:
        print('###Checking accuracy on test set###')
    
    num_correct = 0
    num_samples = 0
    correct_each_class = [0, 0, 0, 0, 0]
    num_each_class = [0, 0, 0, 0, 0]
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            labels = labels.to(device=device, dtype=torch.int64)
            
            scores = model(imgs)
            _, preds = scores.max(1)
            
            npPreds = preds.cpu().numpy()
            npLabels = labels.cpu().numpy()
            
            num_samples += imgs.shape[0]
            for i in range(imgs.shape[0]):
                if npPreds[i] == npLabels[i]:
                    num_correct += 1
                    correct_each_class[npLabels[i]] += 1
                    num_each_class[npLabels[i]] += 1
                else:
                    num_each_class[npLabels[i]] += 1
                    
        acc = float(num_correct) / num_samples
        acc_0 = float(correct_each_class[0]) / num_each_class[0]
        acc_1 = float(correct_each_class[1]) / num_each_class[1]
        acc_2 = float(correct_each_class[2]) / num_each_class[2]
        acc_3 = float(correct_each_class[3]) / num_each_class[3]
        acc_4 = float(correct_each_class[4]) / num_each_class[4]
        
        if train:
            history["train_acc"].append(acc)
            history["train_0_acc"].append(acc_0)
            history["train_1_acc"].append(acc_1)
            history["train_2_acc"].append(acc_2)
            history["train_3_acc"].append(acc_3)
            history["train_4_acc"].append(acc_4)
            print('Total: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print('Class 0: got %d / %d correct (%.2f)' % (correct_each_class[0], num_each_class[0], 100 * acc_0))
            print('Class 1: got %d / %d correct (%.2f)' % (correct_each_class[1], num_each_class[1], 100 * acc_1))
            print('Class 2: got %d / %d correct (%.2f)' % (correct_each_class[2], num_each_class[2], 100 * acc_2))
            print('Class 3: got %d / %d correct (%.2f)' % (correct_each_class[3], num_each_class[3], 100 * acc_3))
            print('Class 4: got %d / %d correct (%.2f)' % (correct_each_class[4], num_each_class[4], 100 * acc_4))
        elif val:
            history["val_acc"].append(acc)
            history["val_0_acc"].append(acc_0)
            history["val_1_acc"].append(acc_1)
            history["val_2_acc"].append(acc_2)
            history["val_3_acc"].append(acc_3)
            history["val_4_acc"].append(acc_4)
            print('Total: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print('Class 0: got %d / %d correct (%.2f)' % (correct_each_class[0], num_each_class[0], 100 * acc_0))
            print('Class 1: got %d / %d correct (%.2f)' % (correct_each_class[1], num_each_class[1], 100 * acc_1))
            print('Class 2: got %d / %d correct (%.2f)' % (correct_each_class[2], num_each_class[2], 100 * acc_2))
            print('Class 3: got %d / %d correct (%.2f)' % (correct_each_class[3], num_each_class[3], 100 * acc_3))
            print('Class 4: got %d / %d correct (%.2f)' % (correct_each_class[4], num_each_class[4], 100 * acc_4))
        else:
            print('Total: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print('Class 0: got %d / %d correct (%.2f)' % (correct_each_class[0], num_each_class[0], 100 * acc_0))
            print('Class 1: got %d / %d correct (%.2f)' % (correct_each_class[1], num_each_class[1], 100 * acc_1))
            print('Class 2: got %d / %d correct (%.2f)' % (correct_each_class[2], num_each_class[2], 100 * acc_2))
            print('Class 3: got %d / %d correct (%.2f)' % (correct_each_class[3], num_each_class[3], 100 * acc_3))
            print('Class 4: got %d / %d correct (%.2f)' % (correct_each_class[4], num_each_class[4], 100 * acc_4))

#%% Train
def train_model(model, optimizer, loader, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (imgs, labels) in enumerate(loader):
            model.train()  # put model to training mode
            imgs = imgs.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            labels = labels.to(device=device, dtype=torch.int64)

            scores = model(imgs)
            loss = F.cross_entropy(scores, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            #if t % 100 == 0:
            #    print('Iteration %d, loss = %.4f' % (t, loss.item()))

        print('Epoch %d, loss = %.4f' % (e, loss.item()))
        history["train_loss"].append(loss.item())
        check_accuracy(trainLoader, model, True, False, False)
        check_accuracy(valLoader, model, False, True, False)
        print()

#%% plot
def plotAcc(modeltype="swin-T"):
    # Rebuild the matplotlib font cache
    #fm._rebuild()
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'DejaVu Sans' # font
    plt.rcParams['font.size'] = 18         # font size
    plt.rcParams['axes.linewidth'] = 2     # axes width

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0, 0, 1, 1])    # Add axes object to our figure that takes up entire figure
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.plot(history["train_acc"], linewidth=2, color='b', label="Train Acc", alpha = 1)
    ax.plot(history["val_acc"], linewidth=2, color='r', label="Val Acc", alpha = 1)
    train_bacc = (np.array(history["train_ctrl_acc"]) + np.array(history["train_mci_acc"]) + 
                 np.array(history["train_hiv_acc"]) + np.array(history["train_hand_acc"])) / 4
    val_bacc = (np.array(history["val_ctrl_acc"]) + np.array(history["val_mci_acc"]) + 
                 np.array(history["val_hiv_acc"]) + np.array(history["val_hand_acc"])) / 4
    ax.plot(train_bacc, linewidth=2, color='c', label="Train BAcc", alpha = 1)
    ax.plot(val_bacc, linewidth=2, color='k', label="Val BAcc", alpha = 1)
    ax.set_ylabel('Accuracy', labelpad=10, fontsize=20)
    ax.set_xlabel('Epochs', labelpad=10, fontsize=20)
    ax.grid(color='g', ls = '-.', lw = 0.5)
    plt.legend(loc="upper left", fontsize=20)
    plt.title(modeltype + " Accuracy History")
    plt.savefig("Figure1_"+modeltype+".png", dpi=300, transparent=False, bbox_inches='tight')
    plt.show()

def plotLoss(modeltype="swin-T"):
    # Rebuild the matplotlib font cache
    #fm._rebuild()
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'DejaVu Sans' # font
    plt.rcParams['font.size'] = 18         # font size
    plt.rcParams['axes.linewidth'] = 2     # axes width

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0, 0, 1, 1])    # Add axes object to our figure that takes up entire figure
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.plot(history["train_loss"], linewidth=2, color='b', label="Train loss", alpha = 1)
    ax.set_ylabel('Loss', labelpad=10, fontsize=20)
    ax.set_xlabel('Epochs', labelpad=10, fontsize=20)
    ax.grid(color='g', ls = '-.', lw = 0.5)
    plt.legend(loc="upper right", fontsize=20)
    plt.title(modeltype + " Loss History")
    plt.savefig("Figure2_"+modeltype+".png", dpi=300, transparent=False, bbox_inches='tight')
    plt.show()

#%% main
if __name__ == '__main__':
    model = swin_t()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    train_model(model, optimizer, trainLoader, epochs=args.epochs)
    check_accuracy(testLoader, model, False, False, True)
    plotAcc(modeltype="swin-T")
    plotLoss(modeltype="swin-T")
