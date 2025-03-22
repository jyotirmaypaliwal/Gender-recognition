# For data handling
import numpy as np
import pandas as pd

# For plotting data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Libraries to interact with the OS
import os
import sys

#%matplotlib inline
#%pylab inline

# Main library
import torch 

# For image processing
import torchvision 
import torchvision.transforms as transforms 
from torchvision.io import read_image

# For data handling and batch processing
from torch.utils.data import Dataset ,DataLoader

#Others
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F



# Variables
pd.options.mode.chained_assignment = None
images_attrr = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')
images_attr = images_attrr.iloc[0:15000,:]
img_fol = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'


images_attr.replace(to_replace=-1, value=0, inplace=True)



# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        imagee = read_image(img_path)
        image = self.transform(imagee)
        label = self.img_labels.iloc[idx, 21]
        return image, label
    
    transform = transforms.Compose([transforms.ToPILImage(),
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])


# Loading data
tester = CustomImageDataset(images_attr, img_fol)
train_dataloader = DataLoader(tester, batch_size=1000, shuffle=True )

# Splitting data into training and test
train_length = 10000
test_length = 5000

train_dataset,test_dataset=torch.utils.data.random_split(tester,(train_length,test_length))

dataloader_train=torch.utils.data.DataLoader(train_dataset,batch_size=200, shuffle=True)

dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True)



## Setting up our device(cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing and customizing model
from torchvision import models
model = models.inception_v3(pretrained=True)
for param in model.parameters():
    param.requires_grad=False
    
model.AuxLogits.fc = nn.Sequential(nn.Linear(768, 1),
                                   nn.Sigmoid())
model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1),
                        nn.Sigmoid())


# More device configurations
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Optimizer
import torch.optim 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss
def compute_loss(y_hat, y):
    return F.binary_cross_entropy(y_hat, y)


# Training loop
for epoch in range(20):
    total_loss = 0
    
    for batch in dataloader_train:
        image, label = batch
        label = torch.unsqueeze(label,1).float()

        image = image.to(device=device)
        label = label.to(device=device)

        
        outputs, aux_outputs = model(image)
        loss = compute_loss(outputs, label) + (0.4 * (compute_loss(aux_outputs, label)))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print("epoch: ", epoch, " loss: ", total_loss)



# Validation loop
tl=0
total_correct=0
for batch in dataloader_test:
    image, label = batch
    label = torch.unsqueeze(label,1).float()
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.to(device);
    image = image.to(device=device)
    label = label.to(device=device)
    
    outputs, aux_outputs = model(image)
    loss = compute_loss(outputs, label) 
    
    tl += loss.item()
    
    
    
    l = np.array([[]])
    for output in outputs:
        if output >= 0.5:
            output =1
            l=np.append(l,output)
        else:
            output=0
            l=np.append(l,output)
            
    l = torch.from_numpy(l)
    l = torch.unsqueeze(l,1).float().to(device=device)
    
      
    
    
    def correct_pred(output, label):
        return output.eq(label).sum().item()
    
    
    
    total_correct += correct_pred(l, label)
    
    
    
print("Total loss: ", tl, " Total correct: ", total_correct, " 5000")


accuracy = ((total_correct*100)/5000)
print("Our accuracy percent is: ", accuracy)