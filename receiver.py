from tensorflow import keras
import cv2
import sys
import string
import numpy as np

from dash_canvas.utils import array_to_data_url, parse_jsonstring

#from src.predict import *

shape = (200,800)
chars = '0123456789' + string.ascii_lowercase

#model = keras.models.load_model('models/draw_model.h5')
model = keras.models.load_model('models/4cnn_draw_model.h5')
#model = keras.models.load_model('models/fin_draw_model.h5')

print('receiving string with length:', len(sys.argv[1]))

# Get image string from app.py
string = sys.argv[1]

# Convert image string to numpy array
mask = parse_jsonstring(string, shape)
img = (255 * mask).astype(np.uint8)
img = cv2.resize(img, (200,50), interpolation = cv2.INTER_AREA)
 

# Use model to predict image
img = img / 255.0
onehotpred = np.array(model.predict(img.reshape(1,50,200,1))).reshape(5,36)    
pred = ''

# Find prediction for each character output
for i in onehotpred:
    c = chars[np.argmax(i)]
    pred += c

# Return prediction on app.py
print(pred)
 
    
# for without line
"""    
# split the drawn image
split_img = split_drawn(img)
    
# predict each character
pred_arr = model.predict(split_img)
    
pred = ''
for i in pred_arr:
    c = chars[np.argmax(i)]
    pred += c
    
print(str(img.shape))
print(str(split_img.shape))

#for i in split_img:
    #char_img = array_to_data_url((i.reshape(28,28).astype(np.uint8)))
    #print(char_img)

print(pred)

"""