##############Set up##############
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
from torch.nn.modules.activation import CELU
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T
from tqdm import tqdm
#from util import check_accuracy_part34, train_part34, Flatten

import numpy as np
import matplotlib.pyplot as plt

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)


##############Load Dataset##############
PATH_OF_DATA = '/home/ubuntu/dataset/'
data_transforms = T.Compose([
                    T.CenterCrop(1200),
                    T.ToTensor(),
                    #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
image_datasets = dset.ImageFolder(root=PATH_OF_DATA, transform=data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=16, shuffle=True, num_workers=2)

#Split data into training, validation and test with proportion 8:1:1
total_size = len(image_datasets)
training_size = int(total_size * 0.8)
validation_size = int(total_size * 0.1)
test_size = total_size - training_size - validation_size
training, validation, test = torch.utils.data.random_split(image_datasets, [training_size, validation_size, test_size])

#Size all these data for more efficient dev cycle
# training_half1, _ = torch.utils.data.random_split(training, [training_size//10, training_size - training_size // 10])
# validation_half1, _ = torch.utils.data.random_split(validation, [validation_size//10, validation_size - validation_size // 10])

#Load data with dataloaders, define batch_size here
trainingLoaders = torch.utils.data.DataLoader(training, batch_size=16, shuffle=True)
validationLoaders = torch.utils.data.DataLoader(validation, batch_size=16, shuffle=True)
testLoaders = torch.utils.data.DataLoader(test, batch_size=16, shuffle=True)

# trainingHalf1Loaders = torch.utils.data.DataLoader(training_half1, batch_size=4, shuffle=True)
# validationHalf1Loaders = torch.utils.data.DataLoader(validation_half1, batch_size=4, shuffle=True)

print("Training Data Length: ", len(training))
print("Validation Data Length: ", len(validation))
print("Test Data Length: ", len(test))

# print("TrainingHalf1 Data Length: ", len(training_half1))
# print("ValidationHalf1 Data Length: ", len(validation_half1))


##############Train##############
train_losses = []
val_acc = []

def check_accuracy_part34(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            #x, y = x.cuda(), y.cuda() 
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        val_acc.append(100 * acc)
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part34(model, optimizer, epochs=1):
    """
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(tqdm(trainingLoaders)):
        # for t, (x, y) in enumerate(tqdm(trainingHalf1Loaders)):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                train_losses.append(loss.item())
                check_accuracy_part34(validationLoaders, model)
                # check_accuracy_part34(validationHalf1Loaders, model)
                print()

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

model = None
optimizer = None

C = 5
out1, out2, out3 = 16, 32, 64
in1, in2, in3 = 3, 16, 32
f1, f2, f3 = 3, 3, 3

#input: 3 x 1200 x 1200
conv1 = nn.Sequential(
    nn.Conv2d(in1, out1, kernel_size=f1, padding=1),
    nn.BatchNorm2d(out1),
    nn.ReLU(),           # Out: 1200 x 1200 x 16
    nn.MaxPool2d(2)      # Out: 600 x 600 x 16
)

#input: 600 x 600 x 16
conv2 = nn.Sequential(
    nn.Conv2d(in2, out2, kernel_size=f2, padding=1),
    nn.BatchNorm2d(out2),
    nn.ReLU(),           # Out: 600 x 600 x 32
    nn.MaxPool2d(2)      # Out: 300 x 300 x 32 
)

#input: 300 x 300 x 32   
conv3 = nn.Sequential(
    nn.Conv2d(in3, out3, kernel_size=f3, padding=1),
    nn.BatchNorm2d(out3),
    nn.ReLU(),           # Out: 300 x 300 x 64
    nn.MaxPool2d(2)      # Out: 150 x 150 x 64
)   

fc =  nn.Sequential(
    nn.Dropout(0.4, inplace=True),
    nn.Linear(64*150*150, C)
)

model = nn.Sequential(
    conv1,
    conv2,
    conv3,
    Flatten(),
    fc,
)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_part34(model, optimizer, epochs=1)

torch.save(model, 'model1.pth')

plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Training_loss1.png')

plt.figure(figsize=(10,5))
plt.title("Validation Accuracy")
plt.plot(val_acc, label="val")
plt.xlabel("iterations")
plt.ylabel("Acc")
plt.legend()
plt.savefig('Accuracy1.png')

check_accuracy_part34(tqdm(testLoaders), model)