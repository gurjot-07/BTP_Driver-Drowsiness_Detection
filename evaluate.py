from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import os
from keras.models import load_model
import seaborn as sns
import numpy as np
from pygame import mixer
import time


train = ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation = ImageDataGenerator(rescale=1/255)

valid_image_gen = validation.flow_from_directory('Valid dataset',target_size=(100,100),batch_size=32,color_mode='grayscale',class_mode='binary')

model = load_model('CNN_model_MRL.h5')
print(model.summary())


loss,accuracy = model.evaluate(valid_image_gen)
print("loss:",loss)
print("Accuracy:",accuracy)



