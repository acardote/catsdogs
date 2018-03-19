#Importing libraries

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

from CNN_archs import model_medium, model_AlexNet, model_LeNet5 

import CNN_archs as CNN

print CNN.a

# Load the data
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images =  [TEST_DIR + i for i in os.listdir(TEST_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

# slice datasets for memory efficiency on Kaggle Kernels
train_images = train_dogs[:2000] + train_cats[:2000]

#Preparing the data
rows = 64
columns = 64
channels = 3

def read_image(path):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.resize(img, (rows, columns),interpolation = cv2.INTER_AREA )
	return img

all_images = []
for i in range(0,len(train_images)):
	img = read_image(train_images[i])
	all_images.append(img)
X_train = np.array(all_images)

Y_train = []
for i in train_images:
	if 'dog' in i:
		Y_train.append(1)
	else:
		Y_train.append(0)

#Normalize the data
X_train = X_train / 255.0

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=42)

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model = model_medium

# Compile the model
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 5
batch_size = 150

# Without data augmentation i obtained an accuracy of 0.98114
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
				validation_data = (X_val, Y_val), verbose = 2, callbacks=[learning_rate_reduction])

Y_pred = model.predict(X_val)

confusion_matrix(Y_val, Y_pred.round())


