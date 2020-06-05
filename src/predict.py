import os
import string
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import cv2

# function to split handwritten captcha
def split_drawn(drawn_img):
    thresh = drawn_img.copy()

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    letter_image_regions = []

    # Iterate through the contours
    for contour in contours:
        # Find location of each character
        (x, y, w, h) = cv2.boundingRect(contour)

        """# If width/height is too large to be a one character, split it
        if w / h > 1.5:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:"""
        letter_image_regions.append((x, y, w, h))

    # Sort the characters
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Add each resized character to an array
    char_lst = []
    for i, letter_bounding_box in enumerate(letter_image_regions):
        # Get location of character
        x, y, w, h = letter_bounding_box

        # Extract the character
        letter_image = thresh[y - 2:y + h + 2, x - 2:x + w + 2]
        
        # Resize the character
        resized = cv2.resize(letter_image, (28,28), interpolation = cv2.INTER_AREA)
        resized = resized.reshape(28,28,1)

        char_lst.append(resized)
    
    return np.array(char_lst)

# function to predict drawn captcha
def predict_drawn(model, img):
    chars = '0123456789' + string.ascii_lowercase
    
    # split the drawn image
    split_img = split_drawn(img)
    
    # predict each character
    pred_arr = model.predict(split_img)
    
    pred = ''
    for i in pred_arr:
        c = chars[np.argmax(i)]
        pred += c

    return pred