import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash_canvas import DashCanvas
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url, parse_jsonstring


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
import cv2
import matplotlib.pyplot as plt

from src.predict import *

canvas_height = 200
canvas_width = 800
shape = (200,800)

chars = string.ascii_lowercase + "0123456789"
#model = keras.models.load_model('draw_model.h5')
# -------------------------------------------------------------------- #

# litera, flatly, mintly
external_stylesheets =['https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/litera/bootstrap.min.css']+['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

app.layout = html.Div(children=[
    html.H1(children='Breaking CAPTCHA', 
            style={'text-align': 'center', 'font-size':'70px',
                    #'color': 'darkcyan',
                    'background': '#60A0B5',
                    #'border': '3px solid indigo',
                    'padding': '7px',
                    #font-family: monospace;
                  }),
        
    # Example captcha images
    html.Img(src=app.get_asset_url('2a76a.png'), style={'width': '250px'}),
    html.Img(src=app.get_asset_url('exd3k.png'), style={'width': '250px'}),
    html.Img(src=app.get_asset_url('ky5fm.png'), style={'width': '250px'}),
    html.Img(src=app.get_asset_url('w8bp5.png'), style={'width': '250px'}),
    
    #html.Br(),

    # Dash canvas for drawing captcha
    html.Div([
            html.H2('Draw a CAPTCHA and press Save to see work'),
            DashCanvas(id='canvas',
                       lineWidth=17,
                       lineColor='black',
                       hide_buttons=["zoom", "pan", "line", "pencil", 
                                     "rectangle", "select"],
                       height=200,
                       width=800,
                       ),
    ], style={'display': 'inline-block', 'vertical-align': 'middle'}),
    
    # Image of drawn captcha
    html.Img(id='captcha_img', style={"width":"50%", "height":"50%"}),
    
    # Hidden div inside the app that stores the drawn image
    html.Div(id='drawn_img', style={'display': 'none'}),
    
    html.Br(),
    
    # Break button and loading spinner that becomes prediction
    html.Div([
            html.H5('Run Captcha Breaking Model'),
            dbc.Button('BREAK', id='start', size='lg', block=True, color='info', 
                       n_clicks=0, style={'display':'inline-block', "width": "30rem",
                                          "height":"6rem", 'font-size':'20px'}),
            html.Br(),
            dbc.Spinner(html.Div([html.H1(id='prediction')]),
                       spinner_style={"width":"7rem", "height":"7rem", "color":"info"}),
    ]),

], style={'textAlign': 'center'})


# -------------------------------------------------------------------- #
# ------------------------ CALL BACKS -------------------------------- #

@app.callback([Output('captcha_img', 'src'),
               Output('drawn_img', 'children')],
              [Input('canvas', 'json_data')])
def show_draw(string):
    if not string:
        raise PreventUpdate
        
    mask = parse_jsonstring(string, shape)
    mask = (255 * mask).astype(np.uint8)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    captcha_img = array_to_data_url(mask)
    
    return captcha_img, string


@app.callback(
     Output('prediction','children'),
     #Output('drawn_img', 'children')],
    [
     Input('start', 'n_clicks'),
    ],
    [State('drawn_img', 'children')],
)
def predict(on, string):
    
    if not on or not string:
        raise PreventUpdate
    
    print('sending string with length:',len(string))
    
    cmd = ['python', 'receiver.py'] + [string]
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    outputs = []
    for line in p.stdout.readlines():
        outputs.append(line)
        print(line)
        
    print('Prediction Completed...')    
    
    
    pred = str(outputs[-1])[2:-3]
    pred = 'Predicted Text: ' + pred

    return pred


# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8811)