# catvsdog
The dataset is from https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
The code trains an image classification model using a pre-trained ResNet18 architecture on a dataset with 'train' and 'test' sets. It utilizes PyTorch and torchvision for data handling, model creation, and training. The training loop is executed for 30 epochs and achieves a high accuracy for the training and test sets. Afterward, the code defines a function to preprocess images and uses the trained model to make predictions on sample images ('cat.jpeg', 'dog.jpeg', 'lion.jpeg'), displaying the predicted class and probabilities. 
The model accurately predicted the correct class for the dog and cat but when tested on the picture of the lion it had a probabilty of 50% for a dog and 42% for a cat.
 
