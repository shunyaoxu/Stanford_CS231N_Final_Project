#%% Setup
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
from swin_transformer import *
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import sampler
from sklearn.model_selection import train_test_split
from os import listdir
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
PATH_OF_DATA = '/home/ubuntu/dataset/'
data_transforms = T.Compose([
                    T.CenterCrop(1200),
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
print("Training Data Length: ", len(train))
print("Validation Data Length: ", len(val))
print("Test Data Length: ", len(test))

#%% Check Accuracy
history = {}
history["val_acc"] = []
history["val_ctrl_acc"] = []
history["val_mci_acc"] = []
history["val_hiv_acc"] = []
history["val_hand_acc"] = []
history["train_loss"] = []
history["train_acc"] = []
history["train_ctrl_acc"] = []
history["train_mci_acc"] = []
history["train_hiv_acc"] = []
history["train_hand_acc"] = []
def check_accuracy(loader, model, train=True, val=False, test=False):
    if train:
        print('Checking accuracy on train set')
    elif val:
        print('Checking accuracy on val set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    num_ctrl_correct = 0
    num_ctrl = 0
    num_mci_correct = 0
    num_mci = 0
    num_mci_correct = 0
    num_mci = 0
    num_hiv_correct = 0
    num_hiv = 0
    num_hand_correct = 0
    num_hand = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            npScores = scores.cpu().numpy()
            npY = y.cpu().numpy()
            npScore = np.where(npScores > 0.5, 1, 0)
            for i in range(x.shape[0]):
                if npY[i][0] == 0 and npY[i][1] == 0:
                    num_ctrl += 1
                    if np.array_equal(npScore[i], npY[i]):
                        num_correct += 1
                        num_ctrl_correct += 1
                if npY[i][0] == 1 and npY[i][1] == 0:
                    num_mci += 1
                    if np.array_equal(npScore[i], npY[i]):
                        num_correct += 1
                        num_mci_correct += 1
                if npY[i][0] == 0 and npY[i][1] == 1:
                    num_hiv += 1
                    if np.array_equal(npScore[i], npY[i]):
                        num_correct += 1
                        num_hiv_correct += 1
                if npY[i][0] == 1 and npY[i][1] == 1:
                    num_hand += 1
                    if np.array_equal(npScore[i], npY[i]):
                        num_correct += 1
                        num_hand_correct += 1    
            num_samples += x.shape[0]
        acc = float(num_correct) / num_samples
        acc_ctrl = float(num_ctrl_correct) / num_ctrl
        acc_mci = float(num_mci_correct) / num_mci
        acc_hiv = float(num_hiv_correct) / num_hiv
        acc_hand = float(num_hand_correct) / num_hand
        if train:
            history["train_acc"].append(acc)
            history["train_ctrl_acc"].append(acc_ctrl)
            history["train_mci_acc"].append(acc_mci)
            history["train_hiv_acc"].append(acc_hiv)
            history["train_hand_acc"].append(acc_hand)
            print('Total: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print('CTRL: got %d / %d correct (%.2f)' % (num_ctrl_correct, num_ctrl, 100 * acc_ctrl))
            print('MCI: got %d / %d correct (%.2f)' % (num_mci_correct, num_mci, 100 * acc_mci))
            print('HIV: got %d / %d correct (%.2f)' % (num_hiv_correct, num_hiv, 100 * acc_hiv))
            print('HAND: got %d / %d correct (%.2f)' % (num_hand_correct, num_hand, 100 * acc_hand))
        elif val:
            history["val_acc"].append(acc)
            history["val_ctrl_acc"].append(acc_ctrl)
            history["val_mci_acc"].append(acc_mci)
            history["val_hiv_acc"].append(acc_hiv)
            history["val_hand_acc"].append(acc_hand)
            print('Total: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print('CTRL: got %d / %d correct (%.2f)' % (num_ctrl_correct, num_ctrl, 100 * acc_ctrl))
            print('MCI: got %d / %d correct (%.2f)' % (num_mci_correct, num_mci, 100 * acc_mci))
            print('HIV: got %d / %d correct (%.2f)' % (num_hiv_correct, num_hiv, 100 * acc_hiv))
            print('HAND: got %d / %d correct (%.2f)' % (num_hand_correct, num_hand, 100 * acc_hand))
        else:
            print('Total: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print('CTRL: got %d / %d correct (%.2f)' % (num_ctrl_correct, num_ctrl, 100 * acc_ctrl))
            print('MCI: got %d / %d correct (%.2f)' % (num_mci_correct, num_mci, 100 * acc_mci))
            print('HIV: got %d / %d correct (%.2f)' % (num_hiv_correct, num_hiv, 100 * acc_hiv))
            print('HAND: got %d / %d correct (%.2f)' % (num_hand_correct, num_hand, 100 * acc_hand))

#%% Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, dataset = 'ucsf'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction= 'none').to(device)
#         self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.dataset = dataset

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        
        if inputs.shape[0] == 1:
            return BCE_loss
        
        pt = torch.exp(-BCE_loss)
        F_loss= 0 
               
            
        F_loss_pos = self.alpha * (1-pt[targets==1])**self.gamma * BCE_loss[targets==1]
        F_loss_neg = (1-self.alpha) * (1-pt[targets==0])**self.gamma * BCE_loss[targets==0]
        
        if inputs.shape[0] == 1:
            if F_loss_pos.nelement() > 0:
                return F_loss_pos
            else:
                return F_loss_neg
        
        F_loss += (torch.mean(F_loss_pos)+torch.mean(F_loss_neg))/2

        return F_loss
#criterion = FocalLoss(alpha = 0.5, gamma = 2.0, dataset = 'ucsf')
#criterion = nn.BCELoss()
ucsf_criterion_cd = FocalLoss(alpha = 0.5, gamma = 2.0, dataset = 'ucsf')
ucsf_criterion_hiv = FocalLoss(alpha = 0.5, gamma = 2.0, dataset = 'ucsf')

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
        for t, (x, y) in enumerate(loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            scores = model(x)
            
            #loss = focalLoss(scores, y).to(device)
            #loss = criterion(scores, y)
            pred_cd1 = scores[:,0].unsqueeze(1)
            pred_hiv1 = scores[:,1].unsqueeze(1)
            labels_cd = y[:,0].unsqueeze(1)
            labels_hiv = y[:,1].unsqueeze(1)
            losscd = ucsf_criterion_cd(pred_cd1, labels_cd).to(device)
            losshiv = ucsf_criterion_hiv(pred_hiv1, labels_hiv).to(device)
            xentropy_loss = losscd + losshiv

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            xentropy_loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        
        print('Iteration %d, loss = %.4f' % (e, xentropy_loss.item()))
        history["train_loss"].append(xentropy_loss.item())
        check_accuracy(trainDataloader, model, True, False, False)
        check_accuracy(valDataloader, model, False, True, False)
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

if __name__ == '__main__':
    '''
    model = swin_t()
    learning_rate = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0000001)
    train_model(model, optimizer, trainDataloader, epochs=40)
    check_accuracy(testDataloader, model, False, False, True)
    plotAcc(modeltype="swin-T")
    plotLoss(modeltype="swin-T")
    '''
    model = swin_s()
    learning_rate = 0.000001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0000001)
    train_model(model, optimizer, trainDataloader, epochs=40)
    check_accuracy(testDataloader, model, False, False, True)
    plotAcc(modeltype="swin-S")
    plotLoss(modeltype="swin-S")
