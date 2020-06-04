import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_daq as daq
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
from dash_table import DataTable


from subprocess import call
from dash.exceptions import PreventUpdate

import numpy as np
import string
import pickle
import json

import time
import requests
from skimage import io
from tensorflow import keras


from src.model import captcha_model

img_path = 'images/2a67a.png'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']
canvas_height = 50
canvas_width = 200
shape = (100,400)

chars = string.ascii_lowercase + "0123456789"
#model = pickle.load(open('src/model.pkl', 'rb'))
model = keras.models.load_model('model.h5')



# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

app.layout = html.Div(children=[
    html.H1(children='Breaking Captcha', style={'text-align': 'center'}),

    html.Img(src="https://www.laarchaeology.org/wp-content/uploads/2017/11/science.jpg", style={'width': '250px'}),
    html.Img(src=img_path, style={'width': '250px'}),

    #html.Div([
    html.H6('Draw a 5 Character Captcha and press Save to see work'),
    #html.Div([
    DashCanvas(id='canvas',
               lineWidth=10,
               hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "select"],
               lineColor='black',
               height=50,
               width=400,
               ),
    #], className="five columns"),
    
    html.Img(id='captcha_img'),# height=100, width=400),
    #]), className="five columns"),
    #]),
    
    DataTable(id='canvas-table',
              style_cell={'textAlign': 'left'},
              columns=[{"name": i, "id": i} for i in columns]),
    #html.Div([
    
    html.Div([]),
    
    html.Div([
        html.H4('Run Captcha Breaking Model'),
            html.Button('BREAK', id='start', n_clicks=0)
    ]),
    html.Div([
        html.H4(id='tt')
    ]),
    
    
])


# -------------------------------------------------------------------- #
# ------------------------ CALL BACKS -------------------------------- #

@app.callback(Output('captcha_img', 'src'),
              [Input('canvas', 'json_data')])
def show_draw(string):
    if not string:
        raise PreventUpdate
        
    mask = parse_jsonstring(string, shape)
    captcha_img = array_to_data_url((255 * mask).astype(np.uint8))
    
    return captcha_img

@app.callback(Output('canvas-table', 'data'),
              [Input('canvas', 'json_data')])
def update_data(string):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate
    return data['objects'][1:]


@app.callback(
    Output('tt','children'),
    [Input('start', 'n_clicks'),],
    #[State('booleanswitch', 'on')]
)
def predict(on):
    
    if not on:
        raise PreventUpdate
    
    # X_test = (mask or captcha_img) after resizing and processing
    
    # onehotpred = np.array(model.predict(X_test)).reshape(5,36)
    #pred = ''
    
    #for i in onehotpred:
    #    c = chars[np.argmax(i)]
    #    pred += c
    
    #tt = 'Predicted Captcha: ' + pred
    
    tt = 'abcd'
    
    return tt
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8811)