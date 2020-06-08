import os
import string
import random

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model

import cv2
from PIL import Image

from src.data_processing import *

class captcha_model():

    def __init__(self):
        
        self.X, self.y, self.labels = None, None, None
        
        self.X_train, self.y_train, self.ll_train = None, None, None
        self.X_test1, self.y_test1, self.ll_test1 = None, None, None
        self.X_test2, self.y_test2, self.ll_test2 = None, None, None
        
        self.model = None
        
    def get_data(self, from_src=False):
        '''Load Data using function from data_processing.py
        '''
        self.X, self.y, self.labels = l_data_mixer(from_src=from_src)
        
        self.X_train, self.y_train, self.ll_train = self.X[:77777], self.y[:, :77777], self.labels[:77777]
        self.X_test1, self.y_test1, self.ll_test1 = self.X[77777:93333], self.y[:, 77777:93333], self.labels[77777:93333]
        self.X_test2, self.y_test2, self.ll_test2 = self.X[93333:], self.y[:, 93333:], self.labels[93333:]
        

    def build_model(self, fc=[32,64,32,16], fs = [5,5,5,5]):
        '''Create model with CNN layers and 5 Outputs
        '''
        input_shape = (50, 200, 1)
        num_chars = 36

        # Input layer
        imgm = Input(shape=input_shape)

        # Convolution and Max Pooling layers
        cn1 = Conv2D(fc[0], (fs[0], fs[0]), padding='same', activation='relu')(imgm)
        mp1 = MaxPooling2D(padding='same')(cn1)  

        cn2 = Conv2D(fc[0], (fs[1], fs[1]), padding='same', activation='relu')(mp1)
        mp2 = MaxPooling2D(padding='same')(cn2)  

        cn3 = Conv2D(fc[0], (fs[2], fs[2]), padding='same', activation='relu')(mp2)
        mp3 = MaxPooling2D(padding='same')(cn3)   

        cn4 = Conv2D(fc[0], (fs[3], fs[3]), padding='same', activation='relu')(mp3)
        mp4 = MaxPooling2D(padding='same')(cn4)  

        # Flatten and Dense layers
        flat = Flatten()(mp4)
        output = []
        for _ in range(5):
            dens1 = Dense(64, activation='relu')(flat)
            drop = Dropout(0.5)(dens1)
            result = Dense(num_chars, activation='sigmoid')(drop)

            output.append(result)

        # Compile model    
        model = Model(imgm, output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        self.model = model
        
    def fit_train(self, batch_size=64, epochs=10, verbose=0, validation_split=0.2):
        # Train Model
        history = self.model.fit(self.X_train, [self.y_train[0], self.y_train[1], self.y_train[2],
                                           self.y_train[3], self.y_train[4]],
                                 batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 validation_split=validation_split)
        
        scores = [max(i).round(3) for i in history.history.values()]
        train_acc = scores[6:11]
        val_acc = scores[-5:]
        print(f'Train Acc: {train_acc}')
        print(f'Val Acc: {val_acc}')
        
        self.model = self.model
    
    def fit_all(self, batch_size=64, epochs=10, verbose=0, validation_split=0.2):
        # Train Model
        history = self.model.fit(self.X, [self.y[0], self.y[1], self.y[2], self.y[3], self.y[4]],
                                 batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 validation_split=validation_split)
        
        # Find accuracy scores to print
        scores = [max(i).round(3) for i in history.history.values()]
        train_acc = scores[6:11]
        val_acc = scores[-5:]
        print(f'Train Acc: {train_acc}')
        print(f'Val Acc: {val_acc}')
        
        self.model = self.model
    
    def save_model(self, model_path):
        self.model.save(model_path)

    def predict_lots(self, X_test, y_test, labels):
                     
        chars = string.ascii_lowercase + "0123456789"

        count = len(X_test)
        correct = 0
        all_preds = np.array(self.model.predict(X_test))

        for i in range(count):
            true = labels[i]

            onehotpred = all_preds[:,i,:]    
            pred = ''

            for i in onehotpred:
                c = chars[np.argmax(i)]
                pred += c

            if true == pred:
                correct += 1

            #print(true, pred)

        return (correct / count)

    def predict_scores(self):
        eval1 = self.model.evaluate(self.X_test1,[self.y_test1[0], self.y_test1[1], self.y_test1[2], self.y_test1[3], self.y_test1[4]], verbose=0)
        eval2 = self.model.evaluate(self.X_test2,[self.y_test2[0], self.y_test2[1], self.y_test2[2], self.y_test2[3], self.y_test2[4]], verbose=0)

        # average loss for each output in the two tests
        out_loss = [i.round(3) for i in np.mean(np.array([eval1[:5], eval2[:5]]), axis=0)]

        # average accuracy for each output in the two tests
        out_acc = [i.round(3) for i in np.mean(np.array([eval1[6:], eval2[6:]]), axis=0)]

        # percent correct in X_test1
        a_acc = self.predict_lots(self.X_test1, self.y_test1, self.ll_test1)
        a_acc = round(a_acc, 3)

        # percent correct in X_test2
        b_acc = self.predict_lots(self.X_test2, self.y_test2, self.ll_test2)
        b_acc = round(b_acc, 3)

        print("Model Scores:")
        print(f"   Average Loss for Outputs: {out_loss}")
        print(f"   Average Acc for Outputs: {out_acc}")
        print(f"   Percent correct in 1st Test: {a_acc}")
        print(f"   Percent correct in 2nd Test: {b_acc}")

    
def main():
    # Create model class
    cap = captcha_model()
    
    # Load data to class
    print('Loading data...')
    cap.get_data()
    print('Data received')
    
    # Build and train model
    print('Building Model...')
    cap.build_model(fc=[32,64,32,16], fs=[7,7,7,7])
    print('Training Model...')
    cap.fit_all()
    
    # Save model
    cap.save_model('cap_model.h5')
    print("Model Saved")

if __name__ == "__main__":
    main()    
    