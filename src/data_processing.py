import os
import string
import random

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

import cv2
from PIL import Image

chars = string.ascii_lowercase + "0123456789"
num_chars = len(chars)

# image data processing functions

def remove_img_border(img):
    # remove border
    img[[0,49],1:199] -= 255
    img[:,0] -= 255
    img[:,199] -= 255
    
    return img
    
def l_data_process(from_src=False):
    '''Processes the l_data folder of 109,053 images 
    Parameters:
    -----------
    from_src: True if function is run from src folder
    
    Returns:
    X: 4 tensor numpy array of image data
    y: One hot encoded targets
    labels: labels for image data
    '''
    if from_src:
        path = '../data/l_data/'
    else:
        path = 'data/l_data/'
        
    X = np.zeros((len(os.listdir(path)), 50, 200, 1)) 
    y = np.zeros((5, len(os.listdir(path)), num_chars)) 
    labels = []
    
    for i, name in enumerate(os.listdir(path)):
        img_path = path + name
        img = cv2.imread(img_path , cv2.IMREAD_UNCHANGED)[:,:,3]
        
        img = remove_img_border(img)
        img = img / 255.0
        img = img.reshape(50, 200, 1)
        
        # Define labels using OneHotEncoding
        label = name[:-4]
        target = np.zeros((5, num_chars))
        for j, l in enumerate(label):
            ind = chars.find(l)
            target[j, ind] = 1
        
        X[i] = img
        y[:, i] = target
        labels.append(label)
   
    return X, y, labels

def s_data_process(from_src=False):
    '''Processes the s_data folder of 1070 images 
    Parameters:
    -----------
    from_src: True if function is run from src folder
    
    Returns:
    X: 4 tensor numpy array of image data
    y: One hot encoded targets
    labels: labels for image data
    '''
    if from_src:
        path = '../data/s_data/'
    else:
        path = 'data/s_data/'
        
    X = np.zeros((len(os.listdir(path)), 50, 200, 1)) 
    y = np.zeros((5, len(os.listdir(path)), num_chars))
    labels = []
    
    for i, name in enumerate(os.listdir(path)):
        img_path = path + name
        img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
        
        # remove background
        flatimg = pd.Series(img.flatten()) 
        flatimg = flatimg.apply(lambda x: 255 if x > 150 else x)
        flatimg = flatimg.apply(lambda x: np.abs(x - 255))
        img = flatimg.values.reshape(50,-1)

        img = img / 255.0
        img = img.reshape(50, 200, 1)
        
        # Define labels using OneHotEncoding
        label = name[:-4]
        target = np.zeros((5, num_chars))
        for j, l in enumerate(label):
            ind = chars.find(l)
            target[j, ind] = 1
        
        X[i] = img
        y[:, i] = target
        labels.append(label)
   
    return X, y, labels

def mix_img(img):
    # random rotate value
    ang_list = [-5,-4,-3,-2,-1,0,0,0,0,0,1,2,3,4,5]
    ang = random.choice(ang_list)
    # rotate image
    im = Image.fromarray(img)
    img = np.asarray(im.rotate(ang))
    
    # random shift value
    shif = np.random.randint(130,200,1)[0]
    # shift image
    img = img[:,:shif]
    size = (200 - shif) * 50
    stacker = np.repeat(0, size).reshape(50,-1)
    img = np.hstack((stacker, img))
    
    return img

def l_data_mixer(from_src=False):
    if from_src:
        path = '../data/l_data/'
    else:
        path = 'data/l_data/'
        
    X = np.zeros((len(os.listdir(path)), 50, 200, 1)) 
    y = np.zeros((5, len(os.listdir(path)), num_chars)) 
    labels = []
    
    for i, name in enumerate(os.listdir(path)):
        img_path = path + name
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,3]
        
        img = remove_img_border(img)
        
        # Mix up image data
        img = mix_img(img)
        
        img = img / 255.0
        img = img.reshape(50, 200, 1)
        
        # Define labels using OneHotEncoding
        label = name[:-4]
        target = np.zeros((5, num_chars))
        for j, l in enumerate(label):
            ind = chars.find(l)
            target[j, ind] = 1
        
        X[i] = img
        y[:, i] = target
        labels.append(label)
   
    return X, y, labels