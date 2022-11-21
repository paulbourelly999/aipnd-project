import matplotlib.pyplot as plt

import torch
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models


# if true set to GPU, if false set to CPU
def set_device(use_gpu):
    #Global Variable for device
    global device
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not use_gpu:
        device = torch.device("cpu")
    else:
        print("GPU is not available, using CPU!")
        device = torch.device("cpu")
    print("Device set to ", device)
        
# Method to create classifier for model
def create_classifier(model_out_features,hidden_units):
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(model_out_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return classifier




# Save the checkpoint of classifier
def save_model(model, hidden_units ,optimizer, model_architecture, learn_rate, directory, epoch, loss):
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'epoch': epoch,
                  'hidden_units' : hidden_units,
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
    model = None
    if checkpoint['model_architecture'] == "densenet_121":
        model = models.densenet121(pretrained=True)
        # Fix base model parameters
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = create_classifier(1024,checkpoint['hidden_units'])
    elif checkpoint['model_architecture'] == "vgg13":
        model = models.vgg13(pretrained=True)
        # Fix base model parameters
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = create_classifier(model.classifier[0].in_features,checkpoint['hidden_units'])
    else:
        print("Model architecture " + checkpoint['model_architecture'] + " NOT supported!")
        return
    
    # Load classifier weights
    model.classifier.load_state_dict(checkpoint['model_state_dict'])
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint["learn_rate"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load training checkpoint information
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    class_to_idx = checkpoint['class_to_idx']
    model.class_to_idx = class_to_idx
    # Move all tensors and model to device
    model.to(device)
    return model,optimizer,epoch,loss



# Process image to be input for model
def process_image(image):
    
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
    
    # Crop out 224X224 center
    left = (image.size[0] - 224)/2
    top = (image.size[1] - 224)/2
    right = (image.size[0] + 224)/2
    bottom = (image.size[1] + 224)/2
    image = image.crop((left,top,right,bottom))
    
    # Convert int (0-255) to float (0-1)
    np_image = np.array(image)
    np_image = np_image.astype('float32')
    np_image /= 255.0

    # Normalize with means and std as follows
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image



## Train model with data and optimizer for provided epochs
def train_model(model, data, epochs, optimizer):
    running_loss = 0
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        for inputs, labels in data:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            print("Training step loss : ", loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training loss for epoch : {running_loss/len(data):.3f}")
        data_val = load_val_data()
        model_eval(model,data_val, criterion)
        print("Epoch ", epoch+1 , " of ", epochs, ".")

           
    print("Training Complete!")
    return model,optimizer,epoch, running_loss



def model_eval(model, data, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
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

    print(f"Validation loss: {test_loss/len(data):.3f}.. "
          f"Validation accuracy: {accuracy/len(data):.3f}")
    model.train()

## Load training data:  include Random flip/rotation/crop in data transform
def load_data(file_path, model):
    data_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    image_dataset = datasets.ImageFolder(file_path, transform=data_transform)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)
    model.class_to_idx= image_dataset.class_to_idx

    return data_loader

# Load validation data
def load_val_data():
    data_transform_test = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    image_dataset = datasets.ImageFolder("flowers/valid/", transform=data_transform_test)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)
    return data_loader


## Create Model given architecture, number of hidden units in hidden layer, learn rate, and label to index map
def create_model(arch,hidden_units,learn_rate):
    model = None
    if arch == "densenet_121":
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = create_classifier(1024,hidden_units)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = create_classifier(model.classifier[0].in_features,hidden_units)
    else:
        print("Model architecture " + arch + " NOT supported!")
        return
    
    
   
    optimizer = optim.Adam(model.classifier.parameters(), learn_rate)
    model.to(device)
    return model,optimizer

def predict(image_path, model, top):
    # Turn model to eval mode
    model.eval()
    # Process image
    image = Image.open(image_path)
    np_image = process_image(image)
    # Convert np array to tensor
    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
    inp = image_tensor.unsqueeze(0)
    # move model and input to device
    inp = inp.to(device)
    model.to(device)
    # feed forward
    ps = torch.exp(model.forward(inp))
    # Top probability and top class
    top_p, top_class = ps.topk(top)
    #Get index to class mapping
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_probs, top_classes =top_p[0], top_class[0]
    top_classes = [idx_to_class[int(cl_index)] for cl_index in top_classes]
    return top_probs, top_classes

def log_prediction(top_p, top_class, category_names):
    
    class_name_to_index_map = None
    with open(category_names, 'r') as f:
        class_name_to_index_map = json.load(f)
    print("Printing class names and calculated probabilities");
    i = 0
    for top_cl in top_class:
        print(class_name_to_index_map[top_cl], " [", top_cl ,"] with probabilty of ", top_p[i].item()*100 ,"%" )
        i +=1


