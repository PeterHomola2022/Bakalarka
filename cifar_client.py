from collections import OrderedDict
import warnings



import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
#from torchsummary import summary
from tqdm import tqdm

import glob
import os



warnings.filterwarnings("ignore", category=UserWarning)
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(str(DEVICE))
print(torch.cuda.get_device_name(0))
best_accuracy = 0

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 47 * 47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1,16 * 47 * 47)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
#model = CNN()
#print(model)
train_losses = []
valid_losses = []

def train(model, train_loader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)   

    for _ in range(epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        valid_loss = 0.0
        
        model.train()
        print("Training-the-model")
        for data, target in tqdm(train_loader):
            # move-tensors-to-GPU 
            data = data.to(DEVICE)
            target = target.to(DEVICE)
        
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            print("WHAT")
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            print("WHAT 2")
            # calculate-the-batch-loss
            loss = criterion(output, target)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * data.size(0)
                
        model.eval()
        print("Validate-the-model")
        for data, target in tqdm(valid_loader):        
            data = data.to(DEVICE)
            target = target.to(DEVICE)       
            output = model(data)        
            loss = criterion(output, target)    
            valid_loss += loss.item() * data.size(0)
            
            # calculate-average-losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(len(valid_losses))

        # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        #if valid lossesgo wrong
        print(len(valid_losses))
        help_number = len(valid_losses)
        if help_number > 1:
            if valid_losses[help_number-1] > valid_losses[help_number-2]:
                print('Without saving finish')
                break


    #saveModel()
    print('Finished Training and Saving Model')

    
def test(model, valid_loader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:     #test_loader
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total



   # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     saveModel()


    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    print("accuracy: "+ str(accuracy) + "; loss: " + str(loss))
    return loss, accuracy   


means = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]

train_transform = transforms.Compose([transforms.Resize((200,200)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means,std)])

test_transform = transforms.Compose([transforms.Resize((200,200)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

valid_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

from torchvision import datasets
def load_data():
    data_dir = './class_dataset/dataset/'
    
    train_loader = datasets.ImageFolder(os.path.join(data_dir,'train'),train_transform)
    valid_datasets = datasets.ImageFolder(os.path.join(data_dir,'val'),valid_transform)
    test_loader = datasets.ImageFolder(os.path.join(data_dir,'test'),valid_transform)
    
    print(train_loader.classes)
    
    train_loader=DataLoader(dataset = train_loader,batch_size=16,shuffle = True, num_workers=4)
    test_loader=DataLoader(dataset = test_loader,batch_size=16,shuffle = False, num_workers=4)
    valid_loader=DataLoader(dataset = valid_datasets,batch_size=16,shuffle = False, num_workers=4)
    num_examples = {"trainset": len(train_loader), "testset": len(test_loader)}
    return train_loader, valid_loader, test_loader, num_examples

def saveModel():
    path = "./model.pth"
    torch.save(model.state_dict(), path)
    

model = CNN().to(DEVICE)
train_loader, valid_loader, test_loader, num_examples = load_data()
list_of_files = [fname for fname in glob.glob("./model*")]
if len(list_of_files) !=0:
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    model.load_state_dict(state_dict)
    


import flwr as fl
class CifarClient(fl.client.NumPyClient):
    print("CONNECTION TO SERVER")
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]



    def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)



    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, train_loader, epochs=10)
        return self.get_parameters(config={}), len(train_loader.dataset), {}



    def evaluate(self, parameters, config):
            global best_accuracy
            self.set_parameters(parameters)
            loss, accuracy = test(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                path = './model.pth'
                torch.save(model.state_dict(), path)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

        
   # Start client
fl.client.start_numpy_client(
    server_address="147.232.60.145:8080",
    client=CifarClient(),)
#loss, accuracy = test(net, valid_loader)
#print("Loss: ", loss)
#print("Accuracy: ", accuracy)     