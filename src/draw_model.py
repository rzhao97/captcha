import os
import string
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model

from PIL import Image
import cv2

from emnist import list_datasets
from emnist import extract_training_samples, extract_test_samples


def remove_upper(X_old, labels):
    # Labels' number regions
    nums = list(range(0, 10))
    upper = list(range(10, 36))
    lower = list(range(36, 62))
    X = []
    y = []
    for x, lab in zip(X_old, labels):
        # Save X data and labels for numbers
        if lab in nums:
            X.append(x)
            y.append(lab)
        
        # Ignore X data and labels for uppercase letters
        elif lab in upper:
            continue
        
        # Save X data and alter labels for lowercase letters
        elif lab in lower:
            lab -= 26
            X.append(x)
            y.append(lab)
    
    return np.array(X), np.array(y)

def load_data():
    X_train, train_labels = extract_training_samples('byclass')
    X_test, test_labels = extract_test_samples('byclass')
    
    X_train, train_labels = remove_upper(X_train, train_labels)
    X_test, test_labels = remove_upper(X_test, test_labels)
    
    chars = '0123456789' + string.ascii_lowercase
    num_chars = len(chars)
    
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
        
    return X_train, X_test, train_labels, test_labels

def build_model():
    
    X_train, X_test, train_labels, test_labels = load_data()
    
    chars = '0123456789' + string.ascii_lowercase
    num_chars = len(chars)
    
    input_shape = (28, 28, 1)

    # Input layer
    imgm = Input(shape=input_shape)

    # Convolution and Max Pooling layers
    cn1 = Conv2D(16, (5,5), padding='same', activation='relu')(imgm)
    mp1 = MaxPooling2D(padding='same')(cn1)  

    cn2 = Conv2D(32, (5,5), padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(padding='same')(cn2)  

    cn3 = Conv2D(16, (5,5), padding='same', activation='relu')(mp2)
    mp3 = MaxPooling2D(padding='same')(cn3)   

    # Flatten and Dense layers
    flat = Flatten()(mp3)
    dens1 = Dense(128, activation='relu')(flat)
    #drop = Dropout(0.5)(dens1)
    bn = BatchNormalization()(dens1)
    output = Dense(num_chars, activation='softmax')(bn)#(drop)

    # Compile model
    model = Model(imgm, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    # Train model
    history = model.fit(X_train, train_labels, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    
    return model
    