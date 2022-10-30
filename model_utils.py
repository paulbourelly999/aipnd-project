#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports here
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models


# In[2]:


# Method to create classifier for model
def create_classifier(model_out_features,hidden_units):
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(model_out_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(0.2)),
                              ('fc3', nn.Linear(hidden_units, hidden_units)),
                              ('relu3', nn.ReLU()),
                              ('dropout3', nn.Dropout(0.2)),
                              ('fc4', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return classifier


# In[3]:


# Save the checkpoint of classifier
def save_model(model,optimizer, model_architecture, learn_rate, directory, epoch, loss):
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'epoch': epoch,
                  'model_state_dict': model.classifier.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,
                  'model_architecture': model_architecture,
                  'learn_rate': learn_rate
                 }

    torch.save(checkpoint, directory + 'checkpoint.pth')


# In[4]:


def load_model(filepath):
    checkpoint = torch.load(filepath)
    # Download base model
    model
    if checkpoint['model_architecture'] == "densenet_121":
        model = models.densenet121(pretrained=True)
    elif checkpoint['model_architecture'] == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        print("Model architecture " + checkpoint['model_architecture'] + " NOT supported!")
        return
    # Fix base model parameters
    for param in model.parameters():
        param.requires_grad = False
    classifier = create_classifier()
    # Load classifier weights
    classifier.load_state_dict(checkpoint['model_state_dict'])
    model.classifier = classifier
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint["learn_rate"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load training checkpoint information
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    class_to_idx = checkpoint['class_to_idx']
    model.class_to_idx = class_to_idx
    # Move all tensors and model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss.to(device)
    return model,optimizer,epoch,loss


# In[5]:


# Process image to be input for model
def process_image(image):
    print("Image size :", image.size)
    
    # Resize image where shortest side is 256
    if ( image.size[0] > image.size[1]):
        resize = image.size[0], 256
        image.thumbnail(resize)
    elif ( image.size[1] > image.size[0]):
        resize = 256, image.size[1]
        image.thumbnail(resize)
    else:
        resize  = 256, 256
        image.thumbnail(resize)
    print("Image resized : " , image.size)
    
    # Crop out 224X224 center
    left = (image.size[0] - 224)/2
    top = (image.size[1] - 224)/2
    right = (image.size[0] + 224)/2
    bottom = (image.size[1] + 224)/2
    image = image.crop((left,top,right,bottom))
    print("Image cropped size ", image.size)
    
    # Convert int (0-255) to float (0-1)
    np_image = np.array(image)
    print('Data Type: %s' % np_image.dtype)
    print('Min: %.3f, Max: %.3f' % (np_image.min(), np_image.max()))
    np_image = np_image.astype('float32')
    np_image /= 255.0
    print('Min: %.3f, Max: %.3f' % (np_image.min(), np_image.max()))

    # Normalize with means and std as follows
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    print('Min: %.3f, Max: %.3f' % (np_image.min(), np_image.max()))
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


# In[ ]:


## Train 
def train_model(model, data, epochs, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 0
    running_loss = 0
    print_every = 20
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        for inputs, labels in data:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            #print("Log PS : " , logps)
            loss = criterion(logps, labels)
            print("Loss : ", loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if  steps % 20 == 0:
                data_val = load_val_data()
                model_eval(model,data_val, criterion)
                running_loss = 0



        print("Epoch ", epoch , " of ", epochs, ".")

           
    print("Training Complete!")
    return model,optimizer,epoch, running_loss


# In[ ]:

def model_eval(model, data, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(data):.3f}.. "
          f"Test accuracy: {accuracy/len(data):.3f}")
    model.train()

## Load data
def load_data(file_path):
    data_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    image_dataset = datasets.ImageFolder(file_path, transform=data_transform)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)
    return data_loader

def load_val_data():
    data_transform_test = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    image_dataset = datasets.ImageFolder("flowers/valid/", transform=data_transform_test)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)
    return data_loader
# In[1]:


## Create Model
def create_model(arch,hidden_units,learn_rate,class_to_index_map):
    model = None
    if arch == "densenet_121":
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        print("Model Classifier : ", model.classifier)
        model.classifier = create_classifier(1024,hidden_units)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        print("Model Classifier : ", model.classifier)
        model.classifier = create_classifier(model.classifier[0].in_features,hidden_units)
    else:
        print("Model architecture " + arch + " NOT supported!")
        return
    
    print(model)
    
   
    model.class_to_idx = class_to_index_map
    optimizer = optim.Adam(model.classifier.parameters(), learn_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model,optimizer


# In[ ]:




