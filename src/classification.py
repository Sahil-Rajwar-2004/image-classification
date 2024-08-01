import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tkinter import filedialog as fd
from PIL import Image
import os


class Nets(nn.Module):
    """
    Custom Neural Network for image classification. 

    The network architecture includes:
    - Three convolutional layers with batch normalization
    - Max pooling layers
    - A dropout layer for regularization
    - Two fully connected layers
    """

    def __init__(self, num_classes=2):
        super(Nets,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=256 * 28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Nets(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CUDA device if available otherwise use CPU
model.to(device)


"""
epoch: tells you the number of iterations that this model gonna train on the data
learning_rate: number of steps that it takes to learn or how it descent gradiently
batch_size: depends on the data use when we use videos or photos
"""
epochs = 150
learning_rate = 0.001
batch_size = 16

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root = "D:\\ibm\\project\\src\\train\\",transform = transform)
train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"epoch: [{epoch+1}/{epochs}], loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(),"model.pth")

model = Nets(num_classes = 2)
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()

images = fd.askopenfilenames(filetypes = [("PNG","*.png"),("JPEG","*.jpeg"),("JPG","*.jpg")])
score = 0
for image in images:
    img = Image.open(image).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _,predicted = torch.max(output,1)

    class_names = ["not a dog","a dog"]
    prediction = class_names[predicted.item()]
    if prediction == "a dog": score += 1
    print(f"{os.path.basename(image)} is : {prediction}")

print(f"score: {score / len(images)}")