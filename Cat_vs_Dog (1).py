
import cv2
import imghdr
import torch
import torch.nn as nn

import os 
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torchvision.datasets as datasets

Train_set = 'train'
Test_set = 'test'
print("Training Classes:", os.listdir(Train_set))
print("Test Classes:", os.listdir(Test_set))


# Count total number of images in test set
total_test_images = sum(len(files) for _, _, files in os.walk(Test_set))
print("Total number of Test images:", total_test_images)


# Count total number of images in training set
total_train_images = sum(len(files) for _, _, files in os.walk(Train_set))
print("Total number of Train images:", total_train_images)

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train = datasets.ImageFolder(Train_set, transform=transform)
test =  datasets.ImageFolder(Test_set, transform=transform)


# In[7]:


batch_size = 16
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

# Define the device (GPU or CPU)
device = ('cuda')

import torchvision.models as models

# Define the model
class ImageClassificationModule(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(ImageClassificationModule, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

# Initialize the model and move it to the device
from torchsummary import summary
model = ImageClassificationModule().to(device)

summary(model, input_size=(3, 224, 224))

import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    # Print training statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {train_loss / len(train_loader):.4f}, '
          f'Training Accuracy: {100 * train_correct / total_train_images:.2f}%')

# Test the model
model.eval()  
test_loss = 0.0
test_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()

# Calculate average test loss
average_test_loss = test_loss / len(test_loader)

# Print test statistics
print(f'Test Loss: {average_test_loss:.4f}, '
      f'Test Accuracy: {100 * test_correct / total_test_images:.2f}%')

# save the model
torch.save(model.state_dict(), 'imageclassifiermodel.pth')

# # Deploying the model

from PIL import Image

# Function to load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = Image.fromarray(image) 
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0)  

image_path = 'cat.jpeg'

# Preprocess the image
input_image = preprocess_image(image_path)
input_image = input_image.to(device)


image = cv2.imread('cat.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis('off') 
plt.show()

# Set the model to evaluation mode
model.eval()
# Make predictions
with torch.no_grad():
    output = model(input_image)

# Get the predicted class index
_, predicted_class = torch.max(output, 1)

# Map class index to class labels
class_labels = {0: 'cat', 1: 'dog'}

# Print the predicted class label
predicted_label = class_labels[predicted_class.item()]
print(f'Predicted Class: {predicted_label}')

# Get predicted probabilities in percentage
probabilities = torch.sigmoid(output)
cat_probability = probabilities[0, 0].item() * 100
dog_probability = probabilities[0, 1].item() * 100

# Print predicted probabilities
print(f'Probability for Cat: {cat_probability:.2f}%')
print(f'Probability for Dog: {dog_probability:.2f}%')

image_path = 'dog.jpeg'

# Preprocess the image
input_image = preprocess_image(image_path)
input_image = input_image.to(device)

image = cv2.imread('dog.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis('off') 
plt.show()


# Set the model to evaluation mode
model.eval()
# Make predictions
with torch.no_grad():
    output = model(input_image)

# Get the predicted class index
_, predicted_class = torch.max(output, 1)

# Map class index to class labels
class_labels = {0: 'cat', 1: 'dog'}

# Print the predicted class label
predicted_label = class_labels[predicted_class.item()]
print(f'Predicted Class: {predicted_label}')

# Get predicted probabilities in percentage
probabilities = torch.sigmoid(output)
cat_probability = probabilities[0, 0].item() * 100
dog_probability = probabilities[0, 1].item() * 100

# Print predicted probabilities
print(f'Probability for Cat: {cat_probability:.2f}%')
print(f'Probability for Dog: {dog_probability:.2f}%')

image_path = 'lion.jpeg'

# Preprocess the image
input_image = preprocess_image(image_path)
input_image = input_image.to(device)


image = cv2.imread('lion.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis('off') 
plt.show()

# Set the model to evaluation mode
model.eval()
# Make predictions
with torch.no_grad():
    output = model(input_image)

# Get the predicted class index
_, predicted_class = torch.max(output, 1)

# Map class index to class labels
class_labels = {0: 'cat', 1: 'dog'}

# Print the predicted class label
predicted_label = class_labels[predicted_class.item()]
print(f'Predicted Class: {predicted_label}')

# Get predicted probabilities in percentage
probabilities = torch.sigmoid(output)
cat_probability = probabilities[0, 0].item() * 100
dog_probability = probabilities[0, 1].item() * 100

# Print predicted probabilities
print(f'Probability for Cat: {cat_probability:.2f}%')
print(f'Probability for Dog: {dog_probability:.2f}%')

