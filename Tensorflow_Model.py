import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import time
from os import listdir
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import MaxPooling1D, Conv1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPool2D, Activation

d0 = "/home/ubuntu/dataset/0/"
d1 = "/home/ubuntu/dataset/1/"
d2 = "/home/ubuntu/dataset/2/"
d3 = "/home/ubuntu/dataset/3/"
d4 = "/home/ubuntu/dataset/4/"

all_data = []
all_label = []

filenames = [f for f in listdir(d0)]
for fn in filenames:
    img = cv2.imread(d0 + filenames[i])
    res = cv2.resize(img, dsize=(800, 800), interpolation = cv2.INTER_AREA).reshape(1, 800, 800, 3)
    all_data.append(res)
    all_label.append(0)
    
filenames = [f for f in listdir(d1)]
for fn in filenames:
    img = cv2.imread(d1 + filenames[i])
    res = cv2.resize(img, dsize=(800, 800), interpolation = cv2.INTER_AREA).reshape(1, 800, 800, 3)
    all_data.append(res)
    all_label.append(1)

filenames = [f for f in listdir(d2)]
for fn in filenames:
    img = cv2.imread(d2 + filenames[i])
    res = cv2.resize(img, dsize=(800, 800), interpolation = cv2.INTER_AREA).reshape(1, 800, 800, 3)
    all_data.append(res)
    all_label.append(2)

filenames = [f for f in listdir(d3)]
for fn in filenames:
    img = cv2.imread(d3 + fn)
    res = cv2.resize(img, dsize=(800, 800), interpolation = cv2.INTER_AREA).reshape(1, 800, 800, 3)
    all_data.append(res)
    all_label.append(3)
    
filenames = [f for f in listdir(d4)]
for fn in filenames:
    img = cv2.imread(d4 + fn)
    res = cv2.resize(img, dsize=(800, 800), interpolation = cv2.INTER_AREA).reshape(1, 800, 800, 3)
    all_data.append(res)
    all_label.append(4)

print("number of samples = ", len(all_data))
print(all_data[0].shape)

all_data = np.concatenate(all_data, axis=0)
labels = np.array(all_label)
oneh_labels = np.zeros((labels.size, 5))
oneh_labels[np.arange(labels.size),labels.astype(int)] = 1
print("shape of all samples:", all_data.shape)
print("shape of all labels:", oneh_labels.shape)

mean_image = np.mean(all_data, axis=0)
all_data = all_data - mean_image

X_train, X_test, y_train, y_test = train_test_split(all_data, oneh_labels, test_size=0.2)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5)
print(f"X_train = {X_train.shape}")
print(f"y_train = {y_train.shape}")
print(f"X_dev = {X_dev.shape}")
print(f"y_dev = {y_dev.shape}")
print(f"X_test = {X_test.shape}")
print(f"y_test = {y_test.shape}")


input = keras.Input(shape=(800, 800, 3))

x = Conv2D(filters=64, kernel_size=11, strides=(4,4), padding="same", activation='relu')(input) # 200, 200, 3
x = MaxPooling2D(2)(x) # 100, 100, 3
x = BatchNormalization()(x)

x = Conv2D(filters=128, kernel_size=5, strides=(2,2), padding="same", activation='relu')(x) # 100, 100, 3
x = MaxPooling2D(2)(x) # 50, 50, 3
x = BatchNormalization()(x)

x = Conv2D(256, kernel_size=3, strides=(1,1), padding="same", activation='relu')(x)
x = Conv2D(256, kernel_size=3, strides=(1,1), padding="same", activation='relu')(x)
x = Conv2D(128, kernel_size=3, strides=(1,1), padding="same", activation='relu')(x)

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs=input, outputs=output)

print(model.summary())

#adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-4)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,
                    y_train,
                    epochs=10, batch_size=128,
                    verbose=1,
                    validation_data=(X_dev, y_dev))