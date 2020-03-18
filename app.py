#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 08:59:34 2020

@author: bala
"""
#libraries for the webapp
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

#libraries for web scraping and misc
import warnings
warnings.filterwarnings('ignore')
import requests
from bs4 import BeautifulSoup

#libraries from pyTorch + pre-trained ALBERT model
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

      
###################################################
#layout section
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Markdown('''## Enter full URL for the passage you want to query'''),
    dcc.Input(
        id='url',
        value='https://docs.microsoft.com/en-us/azure/cloud-adoption-framework/',
        type='text',
        style={'width':'65%'}
    ),
    dcc.Markdown('''## Enter your question below'''),
    dcc.Input(
        id='question',
        value='Explain cloud adoption framework',
        type='text',
        style={'width':'50%'}
    ),
    html.Button(id='button', n_clicks=0, children='Submit'),
    dcc.Markdown('''## This is the answer generated'''),
    html.Div(id='answer')
])
########################################################
#callback section
@app.callback(
    Output(component_id='answer', component_property='children'),
    [Input(component_id='button', component_property='n_clicks')],
     [State(component_id='url', component_property='value'),
     State(component_id='question', component_property='value')
     ]
        )
def hfalbertqna (n_clicks, url, question):
    #reading from an URL
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)
    
    passage5 = ''
    blacklist = [
    	'[document]',
    	'noscript',
    	'header',
    	'html',
    	'meta',
    	'head', 
    	'input',
    	'script',
    ]
    
    for t in text:
    	if t.parent.name not in blacklist:
    		passage5 += '{} '.format(t)
    passage = passage5[0:499]
    if (len(passage)<=2000):
          input_dict = tokenizer.encode_plus(question, passage, return_tensors="pt")
          input_ids = input_dict["input_ids"].tolist()
          start_scores, end_scores = model(**input_dict)
    
          all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
          answer = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace('▁', ' ').strip()
    else:
          answer = 'The given passage exceeds the set character limit of 2,000. Please resize'
    return(answer)


########################################################
#main
if __name__ == '__main__':
    app.run_server(debug=True)
