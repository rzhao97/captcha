import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_daq as daq
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
from dash_table import DataTable


import subprocess
from dash.exceptions import PreventUpdate

import numpy as np
import string
import pickle
import json

import time
import requests
from skimage import io
from tensorflow import keras
import matplotlib.pyplot as plt

from src.predict import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']
canvas_height = 50
canvas_width = 200
shape = (100,400)

chars = string.ascii_lowercase + "0123456789"
#model = keras.models.load_model('draw_model.h5')


# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

app.layout = html.Div(id='dark', children=[
    html.H1(children='Breaking Captcha', style={'text-align': 'center'}),
    
    daq.ToggleSwitch(
        id='daq-light-dark-theme',
        label=['Light', 'Dark'],
        style={'width': '250px', 'margin': 'auto'}, 
        value=False
    ),
    
    html.Br(), 
    
    html.Img(src=app.get_asset_url('2a76a.png'), style={'width': '250px'}),
    html.Img(src=app.get_asset_url('exd3k.png'), style={'width': '250px'}),
    html.Img(src=app.get_asset_url('ky5fm.png'), style={'width': '250px'}),
    html.Img(src=app.get_asset_url('w8bp5.png'), style={'width': '250px'}),

    #html.Div([
    html.H6('Draw a 5 Character Captcha and press Save to see work'),
    #html.Div([
    DashCanvas(id='canvas',
               lineWidth=8,
               hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "select"],
               lineColor='black',
               height=100,
               width=400,
               ),
    #], className="five columns"),
    
    html.Img(id='captcha_img'),# height=100, width=400),
    #]), className="five columns"),
    #]),
    
    DataTable(id='canvas-table',
              style_cell={'textAlign': 'left',
                         'display': 'none'
                         },
              columns=[{"name": i, "id": i} for i in columns]),
    
    # Hidden div inside the app that stores the drawn image
    html.Div(id='drawn_img', style={'display': 'none'}),
    
    html.Div([
        html.H4('Run Captcha Breaking Model'),
            html.Button('BREAK', id='start', n_clicks=0)
    ]),
    html.Div([
        html.H4(id='prediction')
    ]),
    
    
])


# -------------------------------------------------------------------- #
# ------------------------ CALL BACKS -------------------------------- #

@app.callback([Output('captcha_img', 'src'),
               Output('drawn_img', 'children')],
              [Input('canvas', 'json_data')])
def show_draw(string):
    if not string:
        raise PreventUpdate
        
    mask = parse_jsonstring(string, shape)
    
    captcha_img = array_to_data_url((255 * mask).astype(np.uint8))
    
    return captcha_img, string

@app.callback(Output('canvas-table', 'data'),
              [Input('canvas', 'json_data')])
def update_data(string):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate
    return data['objects'][1:]


@app.callback(
     Output('prediction','children'),
     #Output('drawn_img', 'children')],
    [
     Input('start', 'n_clicks'),
     Input('drawn_img', 'children')
    ],
)
def predict(on, string):
    
    if not on:
        raise PreventUpdate
    
    print('sending string with length:',len(string))
    
    cmd = ['python', 'receiver.py'] + [string]
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    outputs = []
    for line in p.stdout.readlines():
        outputs.append(line)
        print(line)
        
    print('Call Completed...')    
    pred = str(outputs[-1])[2:-3]

    return pred, None

    """mask = parse_jsonstring(string, shape)
    img = (255 * mask).astype(np.uint8)
    #print(type(img))

    print('About to predict img...')
    split = split_drawn(img)
    pred = model.predict(split)
    #pred = predict_drawn(model, img)
    print('Success')
    #pred = 'abcd'
    
    return str(pred.shape) # str(pred)"""

#-----------------------------------------------------------------
# Dark Theme Callbacks

@app.callback(
    Output('dark', 'style'),
    [Input('daq-light-dark-theme', 'value')]
)
def change_bg(dark_theme):
    if(dark_theme):
        return {'background-color': '#303030', 'color': 'white'}
    else:
        return {'background-color': 'white', 'color': 'black'}

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8811)