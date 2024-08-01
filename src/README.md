# ***Explaination***

## Dog Classifier

This project implements a custom neural network for image classification, specifically to identify whether an image contains a dog or not. The model is trained on a dataset of labeled images and utilizes a convolutional neural network (CNN) architecture.

### Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

### Project Overview

The Dog Classifier is a deep learning project designed to classify images into two categories: "a dog" and "not a dog." The project includes data preprocessing, model training, and inference. The neural network is built using PyTorch, a popular deep learning framework.

## Model Architecture

The `Nets` class defines a custom CNN with the following layers:

1. **Conv1 Layer:** 3 input channels, 64 output channels, 3x3 kernel size, followed by batch normalization and ReLU activation.
2. **Conv2 Layer:** 64 input channels, 128 output channels, 3x3 kernel size, followed by batch normalization and ReLU activation.
3. **Conv3 Layer:** 128 input channels, 256 output channels, 3x3 kernel size, followed by batch normalization and ReLU activation.
4. **Max Pooling Layers:** Reduce the spatial dimensions of the feature maps.
5. **Dropout Layer:** Regularization to prevent overfitting.
6. **Fully Connected Layers:** The final layers that output class scores.

### Forward Method

The `forward` method defines the forward pass of the network. It applies a series of convolutional layers, activation functions, pooling layers, and fully connected layers to the input data.

## Installation

To install the necessary packages and dependencies, run the following commands:

```bash
pip install torch torchvision pillow
```

## Usage
#### Training
***To train the model, adjust the dataset path in the code and run the script. The training process involves the following:***

- Dataset: The dataset is expected to be in a folder structure where images are stored in subfolders named after their respective classes.
- DataLoader: Handles data loading with specified transformations.
- Loss Function: Cross-Entropy Loss is used for classification.
- Optimizer: Adam optimizer with a specified learning rate.

  ```python
  # Specify the path to the training dataset
  train_dataset = ImageFolder(root="path/to/train/dataset", transform=transform)
  ```

## Inference
After training, the model can be used to classify new images. The script allows the user to select images through a file dialog and outputs the classification results.


## Training
To start training, ensure your dataset is correctly organized, and run the script. The training process will iterate over the dataset for a specified number of epochs, adjusting weights using backpropagation.

- Epochs: 150
- Learning Rate: 0.001
- Batch Size: 16

During training, the loss for each epoch is printed, providing insights into the model's learning progress.
