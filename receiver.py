import sys
from time import sleep

from tensorflow import keras
from dash_canvas.utils import array_to_data_url, parse_jsonstring

from src.predict import *

shape = (100,400)
chars = '0123456789' + string.ascii_lowercase
model = keras.models.load_model('draw_model.h5')


print('receiving string with length:', len(sys.argv[1]))

string = sys.argv[1]

mask = parse_jsonstring(string, shape)
img = (255 * mask).astype(np.uint8)
    
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
print(pred)