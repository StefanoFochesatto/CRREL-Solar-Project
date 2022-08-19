import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import normalize
from keras.metrics import MeanIoU
from keras.utils import to_categorical



## Resizing images and defining the number of class. 
SIZE_X = 416 
SIZE_Y = 640
n_classes = 2

## Pulling training images into a python list. 
train_images = []
imagePath = r'/Users/stefanofochesatto/Desktop/Network Test/dataset/train/images'
maskPath = r'/Users/stefanofochesatto/Desktop/Network Test/dataset/train/masks'

for directory_path in sorted(glob.glob(imagePath)):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "*.jpg"))):
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
       
#Convert list into numpy array        
train_images = np.array(train_images)


## Pulling training masks into a python list.
train_masks = [] 
for directory_path in sorted(glob.glob(maskPath)):
    for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.jpg"))):
        mask = cv2.imread(mask_path, 0)
        mask = np.where(mask > 0, 1, 0)
        ## Nearest Exact interpolation is used to keep masks binary. 
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST_EXACT)       
        train_masks.append(mask)
        
## Convert list into numpy array 
train_masks = np.array(train_masks)



## Encoding the segmentation masks into a one-hot tensor
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
print(np.unique(train_masks_encoded_original_shape))
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

## Create a subset of data for quick testing
## 10% for testing and remaining for training
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

## Further split training data into a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.5, random_state = 0)
print("Class values in the dataset are ... ", np.unique(y_train)) 

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

######################################################
################## Model Training ####################
activation='softmax'
LearningRate = 0.0001
optim = keras.optimizers.Adam(LearningRate)

## Defining Loss Function
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.50, 0.50])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

## Defining tracking metrics
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

## Identifying resnet50 backbone, and pulling the necessary preprocessing function. 
BACKBONE1 = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE1)

## Preprocess input. 
X_train1 = preprocess_input(X_train)
X_test1 = preprocess_input(X_test)

## Defining the model. 
model1 = sm.Linknet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

## Compiling the model and printing the summary. 
model1.compile(optim, total_loss, metrics=metrics)
print(model1.summary())

## Model Fitting and Saving. 
history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=8, 
          epochs=100,
          verbose=1,
          validation_data=(X_test1, y_test_cat))

model1.save('linkNet_arch_res50_backbone_100epochs.hdf5')