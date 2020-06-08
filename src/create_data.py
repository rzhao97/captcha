import os
import string
import random
import numpy as np
import cv2

from emnist import extract_training_samples, extract_test_samples

def remove_upper(X_old, labels):
    nums = list(range(0, 10))
    upper = list(range(10, 36))
    lower = list(range(36, 62))
    X = []
    y = []
    for x, lab in zip(X_old, labels):
        if lab in nums:
            X.append(x)
            y.append(lab)
        elif lab in upper:
            continue
        elif lab in lower:
            lab -= 26
            X.append(x)
            y.append(lab)
    
    return np.array(X), np.array(y)

def load_data():
    X_train, train_labels = extract_training_samples('byclass')
    X_test, test_labels = extract_test_samples('byclass')
    
    # remove capital letters
    X_train, train_labels = remove_upper(X_train, train_labels)
    X_test, test_labels = remove_upper(X_test, test_labels)
    
    # merge train and test datasets
    X = np.vstack((X_train, X_test))
    labels = np.hstack((train_labels, test_labels))
    
    return X, labels

def create_drawn_captchas(X, labels):
    chars = '0123456789' + string.ascii_lowercase
    save_path = '../data/drawn/'
    
    while True:
        # choose 5 random characters
        choices = np.random.randint(0, len(labels), 5)
        hwhite = np.zeros((28, 10))
        vwhite = np.zeros((5,160))
        a,b,c,d,e = (i for i in choices)

        # join the 5 characters together
        cat_img = np.concatenate((hwhite, X[a], X[b],X[c], X[d], X[e], hwhite), axis=1)
        cat_img = np.concatenate((vwhite, cat_img, vwhite), axis=0)
        cat_img = cv2.resize(cat_img, (200,50), interpolation = cv2.INTER_AREA)
        cat_img = cv2.threshold(cat_img, 40, 255, cv2.THRESH_BINARY)[1]

        # start and end points for the line
        points= np.random.randint(0, 49, 2)
        start_point = (0, points[0]) 
        end_point = (199, points[1]) 

        color = (255,255,255) 
        thickness = 2

        # Draw a diagonal black line on image
        img = cv2.line(cat_img, start_point, end_point, color, thickness, lineType=8) 

        # save new image with name
        name = ''.join([chars[labels[i]] for i in choices]) + '.png'
        path = os.path.join(save_path, name)

        cv2.imwrite(path, img)
    
def main():
    # Load data to class
    print('Loading data...')
    X, labels = load_data()
    print('Data received')
    
    # Create new drawn captchas
    print('Creating new drawn captchas...')
    create_drawn_captchas(X, labels)

if __name__ == "__main__":
    main()    
    