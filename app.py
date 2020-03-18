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

#opening files from disk
f = open('/home/bala/Documents/Caliber2020/BBC_COVID.txt', 'r')
passage1 = f.read()
f.close()

#reading from an URL
url = 'https://docs.microsoft.com/en-us/azure/cloud-adoption-framework/'
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
        
###################################################
#layout section
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Markdown('''## Choose a passage to work with'''),
    dcc.Dropdown(
        id='passage',
        options=[
            {'label': 'BBC news item on COVID-19, saved as a .txt', 'value': 'BBC'},
            {'label': 'Microsoft Cloud Adoption Framework, previously web scraped', 'value': 'MCAF'}
        ],
        value='BBC',
        style={'width':'65%'}
    ),
    dcc.Markdown('''## Enter your question below'''),
    dcc.Input(
        id='question',
        value='What is India doing to contain the spread of the virus?',
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
     [State(component_id='passage', component_property='value'),
     State(component_id='question', component_property='value')
     ]
        )
def hfalbertqna (n_clicks, passage, question):
  if (passage=='BBC'):
      passage = passage1
  else:
      passage = passage5[0:4000]
  if (len(passage)<4000):
      input_dict = tokenizer.encode_plus(question, passage, return_tensors="pt")
      input_ids = input_dict["input_ids"].tolist()
      start_scores, end_scores = model(**input_dict)

      all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
      answer = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace('▁', ' ').strip()
  else:
      answer = 'The given passage exceeds the set character limit of 4,000. Please resize'
  return(answer)


########################################################
#main
if __name__ == '__main__':
    app.run_server(debug=True)