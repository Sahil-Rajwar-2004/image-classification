# ***Custom Neural Network for Image Classification***

## This Jupyter Notebook deomnstrate the creation, training, and evaluation of a custom neural network using PyTorch. The neural network is designed to classify images into two categories "a dog" or "not a dog". The notebook covers the following steps:

### ***1. Model Definition***
### ***2. Training the Model***
### ***3. Evaluating the Model***


## 1. Model Definition
  ### Imports
  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torchvision.transforms as transforms
  from torch.utils.data import DataLoader
  from torchvision.datasets import ImageFolder
  from tkinter import filedialog as fd
  from PIL import Image
  import os
  ```

  #### These libraries are imported for creating and training the neural network, handling images and managing data.

  ### Neural Architecture

  ```python
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
          super(Nets, self).__init__()
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
  ```

  - Convolution Layers: Extract features from iamges.
  - Batch Normalization: Stabilizes and speeds up training.
  - Dropout Layer: Helps prevent overfitting.
  - Fully Connected Layers: Outputs the classification.


## 2. Training the Model
  ### Setup
  ```python
  model = Nets(num_classes=2)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  ```

    - Device Selection: Uses CUDA if available, otherwise defaults to CPU

   ### Training Parameters
  ```python
  epochs = 150
  learning_rate = 0.001
  batch_size = 16
  ```

  - Epochs: Number of times the entire dataset is passed through the network.
  - Learning Rate: Controls the step size during optimization.
  - Batch Size: Number of samples per gradient update

  ### Data Preparation
  ```python
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])

  train_dataset = ImageFolder(root="D:\\ibm\\project\\src\\train\\", transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  ```

  - Transformations: Resizes images and converts them to tensors.
  - DataLoader: Handles batching and shuffling of training data.

  ### Training Loop
  ```python
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
  ```

  - Training: Iterates over epochs and updates model weights.

  ### Save the Model
  ```python
  torch.save(model.state_dict(),"model.pth")
  ```

## 3. Evaluating the Model
  ## Load the Model
  ```python
  model = Nets(num_classes=2)
  model.load_state_dict(torch.load("model.pth"))
  model.to(device)
  model.eval()
  ```

  ## Image Classification
  ```python
  images = fd.askopenfilenames(filetypes=[("PNG", "*.png"), ("JPEG", "*.jpeg"), ("JPG", "*.jpg")])
  score = 0
  for image in images:
      img = Image.open(image).convert("RGB")
      input_tensor = transform(img).unsqueeze(0).to(device)

      with torch.no_grad():
          output = model(input_tensor)
          _, predicted = torch.max(output, 1)

      class_names = ["not a dog", "a dog"]
      prediction = class_names[predicted.item()]
      if prediction == "a dog": score += 1
      print(f"{os.path.basename(image)} is: {prediction}")

  print(f"score: {score / len(images)}")
  ```

  - Image Selection: Allows the user to select images for classification.
  - Prediction: Outputs the classification result and calculates the accuracy.


## License
### MIT
