import sys
from time import sleep

from dash_canvas.utils import array_to_data_url, parse_jsonstring
from tensorflow import keras
import cv2

from src.predict import *

shape = (200,800)
chars = '0123456789' + string.ascii_lowercase
model = keras.models.load_model('draw_model.h5')


print('receiving string with length:', len(sys.argv[1]))

string = sys.argv[1]

mask = parse_jsonstring(string, shape)
img = (255 * mask).astype(np.uint8)
img = cv2.resize(img, (200,50), interpolation = cv2.INTER_AREA)
    
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

"""for i in split_img:
    char_img = array_to_data_url((i.reshape(28,28).astype(np.uint8)))
    print(char_img)"""

print(pred)