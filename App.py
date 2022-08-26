# ------------------------------
#
# python version: Python 3.9.7
# Developed by Zeynab (Artemis) Mohseni, 
# Spring 2022
#
# ------------------------------

import os
import pathlib

import pandas as pd
import numpy as np
import dask.dataframe as dd

import plotly
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_bio as db
from dash import dcc
from dash import html
from dash import dash_table
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import Memory
from plotly import tools


location = './cachedir'
memory = Memory(location, verbose=0)

# ============= Launch the application ==============
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
color_discrete_map = {'a': '#74C4FF', 'b': '#6CFF84', 'c': '#FF2BAC', 'd':'#FFB92E'}
color_Result = {'Correct': '#2E41FE', 'Incorrect': '#F70000'}
color_tsne_discrete_map = {'1': '#ffed6f','2': '#6a3d9a','3': '#cab2d6','4': '#fb9a99','5': '#a6cee3','6': '#1f78b4','7': '#e31a1c'}
color_sim_level = {'Very low': '#F70000','Low':'#922141', 'High':'#442477', 'Very high':'#2F399C', 'Accurate':'#2E41FE'}
# ======= Import and preprocessing the datasets =======

df1 = pd.read_csv('data/Sample.csv', sep=",", 
    usecols=['User ID', 'Date', 'Date Week','Month', 'Day', 'Hour', 'Question ID', 'Q Cat.', 'User answer', 
    'Correct answer', 'Result', 'Answer duration', 'Class'])
    
df1 = df1.rename(columns={"User ID": "Student ID", "Q Cat.":"Subject number"})
df1 =  df1[(df1['Answer duration'] > 0.01) & (df1['Answer duration'] < 5)].round(3)
df1['User answer'] = df1['User answer'].replace({'a': 1, 'b': 2, 'c': 3, 'd': 4}).astype(int)
df1['Correct answer'] = df1['Correct answer'].replace({'a': 1, 'b': 2, 'c': 3, 'd': 4}).astype(int)
df1['R'] = df1['Result'].replace({'Correct': 1, 'Incorrect': 0})
df1['Student ID'] = 'S'+ df1['Student ID'].astype(str) 

Student_List ={
'Class #1':list(df1['Student ID'][df1['Class'] == 'Class #1'].unique()),
'Class #2':list(df1['Student ID'][df1['Class'] == 'Class #2'].unique()),
'Class #3':list(df1['Student ID'][df1['Class'] == 'Class #3'].unique()),
'Class #4':list(df1['Student ID'][df1['Class'] == 'Class #4'].unique()),
'Class #5':list(df1['Student ID'][df1['Class'] == 'Class #5'].unique()),
'Class #6':list(df1['Student ID'][df1['Class'] == 'Class #6'].unique()),
'Class #7':list(df1['Student ID'][df1['Class'] == 'Class #7'].unique()),
'Class #8':list(df1['Student ID'][df1['Class'] == 'Class #8'].unique()),
}
names = list(Student_List.keys())
colorscale = [0,1]; 

def sim(n1, n2):
    return round((1 - abs(n1 - n2) / (n1 + n2))*100, 3) #round((n2 * 100 /(n1 + n2)), 2)
# ======= Build the banner =======
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                className='sixhalf columns',
                children=[
                    html.H5("Similarity-Based Grouping tool (SBGtool)"),
                ],
            ),
            html.Div(
                className='two columns',
                children=[
                    html.H6("Choose a class name:"),
                ],
            ),
            html.Div(
                className='onethird columns',
                children=[
                    dcc.Dropdown(id = 'opt7', 
                    options=[
                    {'label':name, 'value':name} for name in names], value = list(Student_List.keys())[0],
                    clearable=False,
                    disabled=False
                    ),
                ],
                style={
                        'padding-right': '5px',
                    },   
            ),
            html.Div(
                children=[
                    html.Button(id="learn-more-button", children="LEARN MORE", n_clicks=0),
                    html.Button(id="reset-button", children="Reset", n_clicks=0),
                ],
            ),
        ],
    )

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
            )
        ],
    )

# ======= Generate the "learn more" button =======
def generate_learn_button():
    return html.Div(
        id="markdown",
        className="learn_button",
        children=(
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(
                            children=(
                                """
                                ##### What is Similarity-Based Grouping Tool (SBGTool) about?

                                *SBGTool* is a web-based Visual Learning Analytics (VLA) tool that assists teachers in categorizing students into different groups 
                                based on their similar learning outcomes.  Teachers could use SBGTool to:

                                * Identify the week that has the highest number of interactions for examining the students' engagement
                                * Find the number of students, correct and incorrect answers in a class over a week or an academic year
                                * Find out which subjects are the most challenging and which are the simplest in a class over a week or an academic year
                                * Identify the number of correct and incorrect answers for the most difficult and easiest subject
                                * Recognize the students with the most answers in different performance levels
                                * Find the date, hour, and subject with the most interactions from students
                                * Compare the outcomes of individual students
                                * Determine the maximum and minimum answer times for each subject and each student throughout the period of a week and an academic year
                                * Find the student with the most engagement over the period of a week and an academic year using the tool

                                
                                ##### What does SBGTool show?

                                SBGTool is split into three sections: __key metrics__, __overview__, and __detail__. The *key metrics* section gives broad information 
                                about the dataset to the teachers. By using *overview* that is a timeline-based section, teachers may get a comprehensive 
                                summary of the students' engagements and the number of correct/incorrect answers in different weeks of an academic year. 
                                *Detail* section contains a table, two bar charts, and three tabs with different visualizations. Teachers can use the table to filter and sort
                                the features and extract detailed information about one student, a certain week, a subject, a user answer among the four answer choices, 
                                a correct answer among the four answer choices, result and the maximum and minimum answer durations. 
                                The proposed bar charts in the left-hand side of the tool can be used by teachers to find the percentages of correct and incorrect answers, 
                                and the most difficult and easitest subjects. 
                                Furthermore, the visualizations presented in the three tabs allow teachers to group students based on their similar learning outcomes, 
                                compare the outcomes of two individual students, and find students with similar learning activities. 

                                """
                            )
                        ),
                    ),
                ],
            )
        ),
    )

# ======= Build LEDDisplay =======
def build_LED():
    return html.Div(
        id="LED-container",
        className="twelve columns",
        children=[
            generate_section_banner("Key Metrics"),
            html.Div([
                html.Div(
                    id="N_Student",
                    children=[
                        daq.LEDDisplay(
                            id="operator-Student",
                            label = "No. STU",
                            value= len(df1['Student ID'].value_counts()),
                            color="#2c2c2e",
                            #backgroundColor="white",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-left': '40px',
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                    },
                ),
                html.Div(
                    id="N_Random",
                    children=[
                        daq.LEDDisplay(
                            id="operator_N_Question",
                            label = "No. QNS",
                            value= len(df1['Question ID'].value_counts()),
                            color="#2c2c2e",
                            #backgroundColor="white",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                        'padding-left': '20px',
                    },
                ),
                html.Div(
                    id="N_1",
                    children=[
                        daq.LEDDisplay(
                            id="operator-A1",
                            label = "No. Ans A",
                            value = str(df1['Student ID'][df1['User answer'] == 1].count()),
                            color="#24A2FF",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                        'padding-left': '20px',
                    },
                ),
                html.Div(
                    id="N_2",
                    children=[
                        daq.LEDDisplay(
                            id="operator-A2",
                            label = "No. Ans B",
                            value = str(df1['Student ID'][df1['User answer'] == 2].count()), 
                            color="#31FF52",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                        'padding-left': '20px',
                    },
                ),
                html.Div(
                    id="N_3",
                    children=[
                        daq.LEDDisplay(
                            id="operator-A3",
                            label = "No. Ans C",
                            value = str(df1['Student ID'][df1['User answer'] == 3].count()), 
                            color="#FF29AC",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                        'padding-left': '20px',
                    },
                ),
                html.Div(
                    id="N_4",
                    children=[
                        daq.LEDDisplay(
                            id="operator-A4",
                            label = "No. Ans D", # Mismatch for option 4
                            value = str(df1['Student ID'][df1['User answer'] == 4].count()), 
                            color="#FEB82E",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                    },
                ),
                html.Div(
                    id="Correct",
                    children=[
                        daq.LEDDisplay(
                            id="operator-Correct",
                            label = "No. Correct",
                            value= str(df1['Student ID'][df1['Result'] == 'Correct'].count()),
                            color="#2E41FE",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                    },
                ),
                html.Div(
                    id="Incorrect",
                    children=[
                        daq.LEDDisplay(
                            id="operator-Incorrect",
                            label = "No. Incorrect",
                            value= str(df1['Student ID'][df1['Result'] == 'Incorrect'].count()),
                            color="#F70000",
                            size=24,
                        ),
                    ],
                className='onehalf columns',
                style={
                        'padding-bottom': '10px',
                        'padding-top': '10px',
                    },
                ),
            ], 
            ),
        ],
    )

# ======= LEDDisplay definition =======
def LEDDisplay1(input1, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    return str(len(dff[column].value_counts()))

def LEDDisplay2(input1, reset_click, click_Data, column1, column2):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    return str(dff['Student ID'][dff[column2] == 'a'].count())

def LEDDisplay3(input1, reset_click, click_Data, column1, column2):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    return str(dff['Student ID'][dff[column2] == 'b'].count())

def LEDDisplay4(input1, reset_click, click_Data, column1, column2):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    return str(dff['Student ID'][dff[column2] == 'c'].count())

def LEDDisplay5(input1, reset_click, click_Data, column1, column2):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    return str(dff['Student ID'][dff[column2] == 'd'].count())

def LEDDisplay6(input1, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    return str(dff['Student ID'][dff['Result'] == 'Correct'].count())

def LEDDisplay7(input1, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    return str(dff['Student ID'][dff['Result'] == 'Incorrect'].count())   

def LEDDisplay8(input1, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1

    return str(len(dff[column].value_counts()))

def opt5_update(input1, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    if not column:
        raise PreventUpdate
    else:
        return [{'label': i, 'value': i} for i in dff[column]]

def opt6_update(input1, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    if not column:
        raise PreventUpdate
    else:
        return [{'label': i, 'value': i} for i in dff[column]]


# ======= create components in the layout code =======
def Card(children, **kwargs):
    return html.Section(children, className="card-style")

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        # Marign: TOP, RIGHT, BOTTOM, LEFT
        style={"margin": "10px 15px 10px 15px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "15px",
                	   "margin-right": "15px",},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )

 # ========= Build similarity panel ==========

def build_similarity_panel():
    return html.Div(
        id="top-section-container",
        className="twelve columns",
        children=[
            generate_section_banner("Overview"),
            html.Div([
                html.Div(
                    className='twelve columns',
                    children=[
                        dcc.Loading(id = "loading-icon0", 
                        children=[html.Div(dcc.Graph(id='similarity_plot', style={'height':280}))], 
                        type="circle", color="#2c2c2e"), 
                    ],   
                ),
            ],
            ),
        ],
    )


# ======= Build chart menu and panel =======
def build_chart_menu():
    return dcc.Tabs(
        #className="twelve columns",
        children =[
        dcc.Tab(label='Students\' performance', children=[
            html.Div(
                id="control-chart-container",
                className="twelve columns",
                children=[
                    html.Div([
                        html.Div(
                            className='fourhalf columns',
                            children=[
                            html.Label(["Select a feature:", 
                            dcc.RadioItems(id = 'items', 
                                options=[
                                {'label': 'Student ID', 'value': 'Student ID'},
                                {'label': 'Date', 'value': 'Date'},
                                {'label': 'Day', 'value': 'Day'},
                                {'label': 'Hour', 'value': 'Hour'},
                                {'label': 'Subject', 'value': 'Subject number'},
                                ], 
                                value = 'Student ID',
                                labelStyle={'display': 'inline-block', 'text-align': 'justify', 'margin-right' :'10px'})]),
                            ],   
                        ),
                    ], 
                    style={"margin": "10px 35px 20px 35px"} # Marign: TOP, RIGHT, BOTTOM, LEFT
                    ),
                    html.Div([
                        html.Div(
                            className='twelve columns',
                            children=[
                                dcc.Loading(id = "loading-icon1", 
                                children=[html.Div(dcc.Graph(id='visualizations_plot', style={'height':350}))], type="circle", color="#2c2c2e"), 
                            ],
                        ),
                    ], 
                    ),
                ])
        ], 
        # style = {
        #     'height' : '30px' 
        #     }
        ),
        dcc.Tab(label='Students\' engagement', children=[
            html.Div(
                id="control-chart-container",
                className="twelve columns",
                children=[ 
                    html.Div([
                        html.Div(
                            className='twelve columns',
                            children=[
                            html.Label(["Select a subject number:", 
                            dcc.RadioItems(id = 'items1', 
                                options=[
                                {'label': '1', 'value': '1'},
                                {'label': '2', 'value': '2'},
                                {'label': '3', 'value': '3'},
                                {'label': '4', 'value': '4'},
                                {'label': '5', 'value': '5'},
                                {'label': '6', 'value': '6'},
                                {'label': '7', 'value': '7'},
                                {'label': 'All', 'value': '8'},
                                ], 
                                value = '8',
                                labelStyle={'display': 'inline-block', 'text-align': 'justify', 'margin-right' :'10px'})]),
                            ],   
                        ),
                    ], 
                    style={"margin": "10px 35px 20px 35px"} # Marign: TOP, RIGHT, BOTTOM, LEFT
                    ),
                    html.Div([
                        html.Div(
                            className='twelve columns',
                            children=[
                                dcc.Loading(id = "loading-icon2", 
                                children=[html.Div(dcc.Graph(id='heatmap_plot', style={'height':350}))], type="circle", color="#2c2c2e"), 
                            ],
                        )
                    ], 
                    ),
                ])
        ], 
        # style = {
        #     'height' : '30px' 
        #     }
        ),
        dcc.Tab(label='Comparison', children=[
            html.Div(
                id="control-chart-container",
                className="twelve columns",
                children=[ 
                    html.Div([
                        html.Div(
                            className='six columns',
                            children=[
                                html.Label(["Select student ID X:", dcc.Dropdown(id = 'opt5', 
                                    options=[{'label':name, 'value':name} for name in list(df1['Student ID'].unique())], 
                                    value = list(df1['Student ID'].unique())[267], #192
                                    clearable=False,
                                    )]),
                            ],
                        ),
                        html.Div(
                            className='six columns',
                            children=[
                                html.Label(["Select student ID Y:", dcc.Dropdown(id = 'opt6', 
                                    options=[{'label':name, 'value':name} for name in list(df1['Student ID'].unique())], 
                                    value = list(df1['Student ID'].unique())[192], #201
                                    clearable=False,
                                    )]
                                ),
                            ],   
                        ),
                    ], 
                    style={"margin": "10px 35px 20px 35px"} # Marign: TOP, RIGHT, BOTTOM, LEFT
                    ),
                    html.Div([
                        html.Div(
                            className='six columns',
                            children=[
                                dcc.Loading(id = "loading-icon3", 
                                children=[html.Div(dcc.Graph(id='Parallel_plot', style={'height':350}))], type="circle", color="#2c2c2e"), 
                            ],
                        ),
                        html.Div(
                            className='six columns',
                            children=[
                                dcc.Loading(id = "loading-icon4", 
                                children=[html.Div(dcc.Graph(id='Parallel_plot1', style={'height':350}))], type="circle", color="#2c2c2e"), 
                            ],
                        ),
                    ], 
                    ),
                ])
        ], 
        # style = {
        #     'height' : '30px' 
        #     }
        ),
    ],
    style = {
        'backgroundColor' : 'white',
        }
    ) 
# ======= Build left panel =======
def build_left_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="Table-container",
                children=[
                    generate_section_banner("Detail"),
                	html.Div(id='table1', style={"margin": "20px 15px 0px 15px"}), # Marign: TOP, RIGHT, BOTTOM, LEFT
                    html.Div(children=[
                                dcc.Loading(id = "loading-icon5", children=[html.Div(dcc.Graph(id='userAnswer-bar', style={'height':60}))], type="circle", color="#2c2c2e"),
                            ],
                        ),
                    dcc.Loading(id = "loading-icon6", children=[html.Div(dcc.Graph(id='similarity_Q_Categoriy', style={'height':350}))], type="circle", color="#2c2c2e"), 
                ],
            ),         
        ],
    )
    
# ============== Build similarity  ==============

@app.callback(
    Output("similarity_plot", "figure"),
    [Input("opt7", "value")],
)

def display_similarity_plot(input1):
    if input1:
        dff = df1[df1['Class']== input1]
        dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
        dff['Correct answer'] = dff['Correct answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})

        dff1 = pd.DataFrame(columns=[])
        dff1['Count']=dff['Date Week'].value_counts()
        dff1['Date']=dff['Date Week'].value_counts().index
        dff1['Correct'] = dff['Date Week'][dff['Result'] == 'Correct'].value_counts().astype(str)
        dff1['Incorrect'] = dff['Date Week'][dff['Result'] == 'Incorrect'].value_counts().astype(str)
        dff1['A_1'] = dff['Date Week'][dff['User answer'] == 'a'].value_counts().astype(str)
        dff1['A_2'] = dff['Date Week'][dff['User answer'] == 'b'].value_counts().astype(str)
        dff1['A_3'] = dff['Date Week'][dff['User answer'] == 'c'].value_counts().astype(str)
        dff1['A_4'] = dff['Date Week'][dff['User answer'] == 'd'].value_counts().astype(str)

        dff1['A_1_C'] = dff['Date Week'][dff['Correct answer'] == 'a'].value_counts().astype(str)
        dff1['A_2_C'] = dff['Date Week'][dff['Correct answer'] == 'b'].value_counts().astype(str)
        dff1['A_3_C'] = dff['Date Week'][dff['Correct answer'] == 'c'].value_counts().astype(str)
        dff1['A_4_C'] = dff['Date Week'][dff['Correct answer'] == 'd'].value_counts().astype(str)
        dff1 = dff1.sort_values('Date', ascending=True)

        fig=go.FigureWidget()
        fig.add_trace(go.Scatter(
            name="Ans. A",
            mode="lines+markers", x=dff1.Date, y=dff1['A_1'],
            connectgaps=True,
            yaxis="y",
            #stackgroup='one',
            line=dict(width=1.5, color='#74C4FF'),
            marker=dict(size=4, color='#74C4FF'),
        ))
        fig.add_trace(go.Bar(
            name="Corr Ans. A", 
            y=dff1['A_1_C'],
            x=dff1.Date,
            orientation='v',
            yaxis="y",
            marker=dict(
                color='#78C6FF',
                line=dict(width=1, color='#78C6FF'),
            ),
        ))

        fig.add_trace(go.Scatter(
            name="Ans. B",
            mode="lines+markers", x=dff1.Date, y=dff1['A_2'],
            connectgaps=True,
            yaxis="y",
            #stackgroup='one',
            line=dict(width=1.5, color='#00D923'),
            marker=dict(size=4, color='#00D923'),
        ))
        fig.add_trace(go.Bar(
            name="Corr Ans. B", 
            y=dff1['A_2_C'],
            x=dff1.Date,
            orientation='v',
            yaxis="y",
            marker=dict(
                color='#7CFF91',
                line=dict(width=1, color='#7CFF91'),
            ),
        ))

        fig.add_trace(go.Scatter(
            name="Ans. C",
            mode="lines+markers", x=dff1.Date, y=dff1['A_3'],
            connectgaps=True,
            yaxis="y",
            #stackgroup='one',
            line=dict(width=1.5, color='#FF2BAC'),
            marker=dict(size=4, color='#FF2BAC'),
        ))
        fig.add_trace(go.Bar(
            name="Corr Ans. C", 
            y=dff1['A_3_C'],
            x=dff1.Date,
            orientation='v',
            yaxis="y",
            marker=dict(
                color='#FF2BAC',
                line=dict(width=1, color='#FF2BAC'),
            ),
        ))

        fig.add_trace(go.Scatter(
            name="Ans. D",
            mode="lines+markers", x=dff1.Date, y=dff1['A_4'],
            connectgaps=True,
            yaxis="y",
            #stackgroup='one',
            line=dict(width=1.5, color='#FFAA00'),
            marker=dict(size=4, color='#FFAA00'),
        ))
        fig.add_trace(go.Bar(
            name="Corr Ans. D", 
            y=dff1['A_4_C'],
            x=dff1.Date,
            orientation='v',
            yaxis="y",
            marker=dict(
                color='#FFCE6C',
                line=dict(width=1, color='#FFCE6C'),
            ),
        ))

        fig.add_trace(go.Scatter(
            name="Total",
            mode="markers+text", x=dff1.Date, y=dff1['Count'], 
            marker_color ='white',
            #connectgaps=True,
            #stackgroup='one',
            yaxis="y2",
            marker=dict(size=1, color='white'),

        ))
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="lines+markers", x=dff1.Date, y=dff1['Correct'],
            connectgaps=True,
            #stackgroup='one',
            yaxis="y2",
            line=dict(width=1.5, color='#2E41FE'),
            marker=dict(size=4, color='#2E41FE'),
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="lines+markers", x=dff1.Date, y=dff1['Incorrect'],
            connectgaps=True,
            #stackgroup='one',
            yaxis="y2",
            line=dict(width=1.5, color='#F70000'),
            marker=dict(size=4, color='#F70000'),
        ))

        fig.update_traces(
            hoverinfo="y+x+name",
            #showlegend=True,

        )
        fig.update_layout(
            margin={'t': 50, 'b':50, 'l':0},
            font=dict(size=11),
            xaxis=dict(
                tickfont_size=8,
                rangeslider_visible=True,
                #rangeslider_bgcolor="#E5E5E5",
                rangeslider_bordercolor= "#444",
                rangeslider_thickness=0.15,
                title= "Week number",
                #tickangle=290
            ),
            yaxis=dict(
                tickfont_size=9,
                anchor="x",
                autorange=True,
                domain=[0, 0.75],
                zeroline=False,
                showline=True,
                mirror=True,
                side="left",
                tickmode="auto",
                ticks="",
                tickfont={"color": "#673ab7"},
                type="linear",
                title= "Count"
            ),
            yaxis2=dict(
                tickfont_size=9,
                anchor="x",
                autorange=True,
                domain=[0.76, 1],
                zeroline=False,
                mirror=True,
                showline=True,
                side="left",
                tickmode="auto",
                ticks="",
                tickfont={"color": "red"},
                type="linear",
                title= ""
            ),
        )
        # Update layout
        fig.update_layout(
            #legend_orientation="v", 
            title="Number of student\'s answers and correct answers in different weeks of using the digital learning tool",
            dragmode="zoom",
            hovermode="x unified",
            #legend=dict(traceorder="reversed", title ='Result'),
            height=280,
            template="plotly_white",
            showlegend = False,
            margin={'t': 50, 'b':50, 'r':30, 'l':70},
            )  
        return fig

# ======= Build table =======
def build_table(input1, reset_click, click_Data):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point]
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff1 = dff.iloc[:,[0,2,1,7,8,9,10,11]]
    dff1['User answer'] = dff1['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    dff1['Correct answer'] = dff1['Correct answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    dff1 = dff1.rename(columns={"Answer duration": "Answ. DUR", "Subject number":"Subject number"})
    T = dash_table.DataTable(
            filter_action='native',
            sort_action='native',
            columns=[{"name": i, "id": i} for i in dff1.columns],
            data=dff1.to_dict('records'),
            style_cell={'whiteSpace': 'normal',
            'height': 'auto', 'font_family': 'Arial', 'font_size': '9px','text_align': 'center'},
            style_header={'whiteSpace': 'normal','height': 'auto', "backgroundColor":"rgb(239, 243, 255)"},
            style_data={'width': '41px','maxWidth': '41px','minWidth': '41px','backgroundColor':"white"},
            page_size=13,
            #fixed_rows={'headers': True},
            #style_table={'overflowY': 'auto'},
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Result} = Correct',
                        'column_id': 'Result'
                    },
                    'backgroundColor': '#2E41FE',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Result} = Incorrect',
                        'column_id': 'Result'
                    },
                    'backgroundColor': '#F70000',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Correct answer} = a',
                        'column_id': 'Correct answer'
                    },
                    'backgroundColor': '#74C4FF',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{Correct answer} = b',
                        'column_id': 'Correct answer'
                    },
                    'backgroundColor': '#6CFF84',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{Correct answer} = c',
                        'column_id': 'Correct answer'
                    },
                    'backgroundColor': '#FF2BAC',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{Correct answer} = d',
                        'column_id': 'Correct answer'
                    },
                    'backgroundColor': '#FFB92E',
                    'color': 'black'
                },                
                {
                    'if': {
                        'filter_query': '{User answer} = a',
                        'column_id': 'User answer'
                    },
                    'backgroundColor': '#74C4FF',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{User answer} = b',
                        'column_id': 'User answer'
                    },
                    'backgroundColor': '#6CFF84',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{User answer} = c',
                        'column_id': 'User answer'
                    },
                    'backgroundColor': '#FF2BAC',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{User answer} = d',
                        'column_id': 'User answer'
                    },
                    'backgroundColor': '#FFB92E',
                    'color': 'black'
                },
                
            ]

        )
    return T

#  ======= Build Q_Categoriy similarity plot =======
def Q_Categoriy_plot(input1, reset_click, click_Data):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    dff['Correct answer'] = dff['Correct answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})

    dff3 = pd.DataFrame(columns=[])
    dff3['Total answers']=dff['Subject number'].value_counts()
    dff3['Subject number']=dff['Subject number'].value_counts().index
    dff3['Correct'] = dff['Subject number'][dff['R'] == 1].value_counts()
    dff3['Incorrect'] = dff['Subject number'][dff['R'] == 0].value_counts()

    dff3 = dff3.fillna(0)

    if dff3['Correct'].count != 0:
        dff3['Similarity'] = round((dff3['Correct'] *100)/dff3['Total answers'], 2)
    else:
        dff3['Similarity'] = 0  

    if dff3['Incorrect'].count != 0:
        dff3['Dissimilarity'] = round(100 - dff3['Similarity'], 2)
    else:
        dff3['Dissimilarity'] = 0  

    dff3['Subject number'] = dff3['Subject number'].replace({1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7'})

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff3['Subject number'], 
            y=dff3['Similarity'], 
            texttemplate="%{y} %",
            textposition="inside",
            textangle=270,
            textfont_color="white",
            hovertemplate="<br>".join([
                "%{y}%",
            ]),
            name='Ease', 
            marker_color='#2E41FE'))
    fig.add_trace(
        go.Bar(
            x=dff3['Subject number'], 
            y=dff3['Dissimilarity'], 
            texttemplate="%{y} %",
            textposition="inside",
            textangle=270,
            textfont_color="white",
            hovertemplate="<br>".join([
                "%{y}%",
            ]),
            name='Difficulty', 
            marker_color='#F70000'))
    fig.update_layout(barmode="relative",
        title='Ease & difficulty of different subjects (%)',
        height=350,
        font=dict(size=11),
        xaxis=dict(tickfont_size=8, title= "Subject number", categoryorder='category ascending'),
        yaxis=dict(
            title= "Ease & Difficulty",
            ),
        margin={'t': 60, 'b':85},
        )
    fig.layout.plot_bgcolor  = 'white'
    return fig

# ======= Build parallel Categories Diagram =======
def similarity_parallel (input1, input2, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff2 = dff1[dff1['Date Week'] == point] 
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2
        else:
            dff2 = dff1
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = df1[(df1['Student ID'] == input2)]
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff2 = dff1[dff1['Date Week'] == point]
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2 
        else:
            dff2 = dff1 
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff = dff.iloc[:,[0, 2, 6, 7, 8, 10, 13]]
    dff = dff.sort_values('Result', ascending=False)
    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    colorscale = [0,1];

    if dff['Student ID'].empty:
        fig = go.Figure(
            layout = {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations":[{
                    "text": "Choose a student ID from the drop-down menu",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 15}
                }]
                })
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
    else:
        fig = px.parallel_categories(dff, 
            dimensions=['Student ID', 'Date Week',  'Subject number', 'User answer','Result'],
            color = 'R', 
            range_color = colorscale,
            color_continuous_scale=px.colors.sequential.Bluered_r,
            labels={'R':'Result', 'Subject number': 'Subject Number', 'User answer':'STU Answ.'},
            )
        fig.update_layout(title = 'Number of interactions for student ID {}'.format(input2)+' is: {}'.format(len(dff['Question ID'].value_counts())),
            height = 350, 
            font=dict(size=11), 
            showlegend = False,
            coloraxis_showscale=False,
            margin={'t': 50, 'b':50, 'r':40, 'l':0},
            )

    if not column:
        raise PreventUpdate
    else:
        return fig

def similarity_parallel1 (input1, input2, reset_click, click_Data, column):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff2 = dff1[dff1['Date Week'] == point] 
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2
        else:
            dff2 = dff1
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = df1[(df1['Student ID'] == input2)]
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff2 = dff1[dff1['Date Week'] == point]
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2 
        else:
            dff2 = dff1 
            if input2:
                dff = dff2[(dff2['Student ID'] == input2)]
            else:
                dff = dff2  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff = dff.iloc[:,[0, 2, 6, 7, 8, 10, 13]]
    dff = dff.sort_values('Result', ascending=False)
    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    colorscale = [0,1];

    if dff['Student ID'].empty:
        fig = go.Figure(
            layout = {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations":[{
                    "text": "Choose a student ID from the drop-down menu",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 15}
                }]
                })
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
    else:
        fig = px.parallel_categories(dff, 
            dimensions=['Student ID', 'Date Week',  'Subject number', 'User answer','Result'],
            color = 'R', 
            range_color = colorscale,
            color_continuous_scale=px.colors.sequential.Bluered_r,
            labels={'R':'Result', 'Subject number': 'Subject Number', 'User answer':'STU Answ.'},
            )
        fig.update_layout(title = 'Number of interactions for student ID {}'.format(input2)+' is: {}'.format(len(dff['Question ID'].value_counts())),
            height = 350, 
            font=dict(size=11), 
            showlegend = False,
            coloraxis_showscale=False,
            margin={'t': 50, 'b':50, 'r':40, 'l':0},
            )

    if not column:
        raise PreventUpdate
    else:
        return fig
                   
# ======= Build bar chart =======
def bar1(input1, reset_click, click_Data):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff3 = pd.DataFrame(columns=[])
    dff3['Total answers']=dff['Class'].value_counts()
    dff3['Correct'] = dff['Class'][dff['Result'] == 'Correct'].value_counts()
    dff3['Incorrect'] = dff['Class'][dff['Result'] == 'Incorrect'].value_counts()
    dff3 = dff3.fillna(0)

    if dff3['Correct'].count != 0:
        dff3['Result_c'] = round((dff3['Correct']/dff3['Total answers'] *100), 2)
    else:
        dff3['Result_c'] = 0  

    if dff3['Incorrect'].count != 0:
        dff3['Result_r'] = round(100 - dff3['Result_c'], 2)
    else:
        dff3['Result_r'] = 0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff3['Result_c'], 
            texttemplate="%{x} %",
            textposition="inside",
            textangle=0,
            textfont_color="white",
            hovertemplate="<br>".join([
                "%{x}%",
            ]),
            name='Correct', 
            marker_color='#2E41FE'))
    fig.add_trace(
        go.Bar(
            x=dff3['Result_r'], 
            texttemplate="%{x} %",
            textposition="inside",
            textangle=0,
            textfont_color="white",
            hovertemplate="<br>".join([
                "%{x}%",
            ]),
            name='Incorrect', 
            marker_color='#F70000'))
    fig.update_layout(
        title="Result",
        font=dict(size=11),
        barmode='stack',
        height = 60,
        margin={'t': 25, 'b':0, 'r':10, 'l':25},
        hovermode="closest",
        yaxis=dict(showticklabels=False, title_font=dict(size =9)),
        xaxis=dict(showticklabels=False),
        )
    fig.update_traces(showlegend=False)
    fig.layout.plot_bgcolor  = 'white'
    return fig

# ======= Build heatmap =======
def heatmap_Similarity(input1, input2, reset_click, click_Data):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})

    if input2 == '1':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 1)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            #color_continuous_scale="Viridis_r",
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title =  'Number of answers in subject 1 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '2':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 2)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title= 'Number of answers in subject 2 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '3':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 3)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title=  'Number of answers in subject 3 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '4':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 4)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title= 'Number of answers in subject 4 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '5':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 5)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title= 'Number of answers in subject 5 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '6':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 6)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title= 'Number of answers in subject 6 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '7':
        df2 = dff
        df2 = df2[(df2['Subject number'] == 7)]
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title= 'Number of answers in subject 7 for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    if input2 == '8':
        df2 = dff
        df2 = df2.groupby(['Student ID', 'User answer']).size().unstack().fillna(0)
        df2["sum"] = df2.sum(axis=1)
        df2 = df2.T
        df2 = df2.loc[:, df2.max().sort_values(ascending=False).index]
        df2 = df2.T
        df2 = df2.iloc[: , :-1] #df2= df2.drop(['sum'], 1)
        df2 = df2.T

        fig = px.imshow(df2,
            labels=dict(x="Student ID", y="User answer", color="Count"),
            color_continuous_scale='Blues',
            #color_continuous_scale="Viridis_r",

            origin='lower',
            aspect='auto', #aspect='equal',
            height = 350,
            )
        fig.update_layout(
            title= 'Number of answers in all subjects for {}'.format(len(df2.columns))+ ' students',
            font=dict(size=11),
            margin={'t': 50, 'b':80, 'r':50, 'l':70},
            coloraxis_showscale=False,
            xaxis=dict(tickfont_size=8, rangeslider_visible=True, tickangle=290)
            )

        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

# ======= Build visulaization plots =======
def visualizations_plot(input1, input2, reset_click, click_Data):
    if input1:
        dff1 = df1[df1['Class'] == input1]
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True
                dff = dff1
    else:
        dff1 = df1  
        if click_Data:
            point = click_Data['points'][0]["x"]
            dff = dff1[dff1['Date Week'] == point] 
        else:
            dff = dff1  
        ctx = dash.callback_context
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-button":
                reset = True

    dff['User answer'] = dff['User answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    dff['Correct answer'] = dff['Correct answer'].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    dff1 = dff

    if input2 == 'Student ID':
        dff2 = pd.DataFrame(columns=[])
        dff2['Total answers']=dff1['Student ID'].value_counts()
        dff2['Student ID']=dff1['Student ID'].value_counts().index
        dff2['Correct'] = dff1['Student ID'][dff1['Result'] == 'Correct'].value_counts()
        dff2['Incorrect'] = dff1['Student ID'][dff1['Result'] == 'Incorrect'].value_counts()
        dff2 = dff2.fillna(0)

        if dff2['Correct'].count != 0:
            dff2['Performance (%)'] = round((dff2['Correct'] *100)/dff2['Total answers'])
        else:
            dff2['Performance (%)'] = 0  

        conditions = [
            (dff2['Performance (%)'] > 99),
            (dff2['Performance (%)'] > 85)  & (dff2['Performance (%)'] <= 99),
            (dff2['Performance (%)'] > 65)  & (dff2['Performance (%)'] <= 85),
            (dff2['Performance (%)'] > 50)  & (dff2['Performance (%)'] <= 65),
            (dff2['Performance (%)'] <= 50),
            
            ]
        # create a list of the values we want to assign for each condition
        values = ['Accurate', 'Very high', 'High', 'Low','Very low']
        # create a new column and use np.select to assign values to it using our lists as arguments
        dff2['Level'] = np.select(conditions, values)

        dff2 = dff2 [dff2 ['Performance (%)']>0]
        dff2 = dff2.sort_values(['Performance (%)', 'Level'], ascending=True)

        fig = px.scatter(dff2, 
             x="Student ID",
             y= "Performance (%)", 
             color="Level", 
             size="Total answers",
             #log_y=True, 
             #facet_col="Level",
             title ='Students with different levels of performance',
             color_discrete_map=color_sim_level,
             hover_data=["Student ID", "Performance (%)", "Total answers", "Level"]
            )
        fig.update_layout(
            margin={'t': 50, 'b':80, 'r':140, 'l':90},
            font=dict(size=11), 
            height = 350,
            xaxis=dict(tickfont_size=8,rangeslider_visible=True, title= "Student IDs", tickangle=300, categoryorder='category ascending'),
            yaxis=dict(title= "Performance (%)"),
            template="plotly_white",
            legend=dict(
                traceorder="reversed", 
                title ='Level')
            ),
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hoverinfo="y+x+name",
            #hovertemplate ='Student ID: %{x}<br>Performance: %{y}%',
            showlegend=True,
        )
        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        #fig.update_yaxes(type="log")
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    dff2 = pd.DataFrame(columns=[])
    dff2['D_Count']=dff1['Day'].value_counts()
    dff2['Correct'] = dff1['Day'][dff1['Result'] == 'Correct'].value_counts()
    dff2['Incorrect'] = dff1['Day'][dff1['Result'] == 'Incorrect'].value_counts()
    dff2['Day']=dff1['Day'].value_counts().index
    if input2 == 'Day':
        dff2 = dff2.sort_values('Day', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Total",
            mode="markers+text", x=dff2.Day, y=dff2['D_Count'], 
            marker_color ='white',
            #connectgaps=True,
            #stackgroup='one',
            marker=dict(size=4, color='#2E41FE'),
        ))
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff2.Day, y=dff2['Correct'], 
            marker_color ='#2E41FE',
            connectgaps=True,
            stackgroup='one',
            marker=dict(size=4, color='#2E41FE'),
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff2.Day, y=dff2['Incorrect'], 
            marker_color ='#F70000',
            connectgaps=True,
            stackgroup='one',
            marker=dict(size=4, color='#F70000'),
        ))
        fig.update_layout(
            title="Number of correct/incorrect answers per day",
            xaxis_title="Day",
            yaxis_title="Count",
            height = 350,
            font=dict(size=11),
            hovermode="x unified",
            showlegend = False,
            margin={'t': 50, 'b':80, 'r':50, 'l':80},
            template="plotly_white",
            legend=dict(
                traceorder="reversed",
                title ='User answer',
                font=dict(
                    color="black"
                ),
                bgcolor="white",
                bordercolor="white",
                borderwidth=1
            )
        )
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hoverinfo="y+x+name",
            showlegend=True,
        )
        #fig.update_xaxes(showspikes=True)
        #fig.update_yaxes(showspikes=True)
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    dff2 = pd.DataFrame(columns=[])
    dff2['H_Count']=dff1['Hour'].value_counts()
    dff2['Correct'] = dff1['Hour'][dff1['Result'] == 'Correct'].value_counts()
    dff2['Incorrect'] = dff1['Hour'][dff1['Result'] == 'Incorrect'].value_counts()
    dff2['Hour']=dff1['Hour'].value_counts().index
    if input2 == 'Hour':
        dff2 = dff2.sort_values('Hour', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Total",
            mode="markers+text", x=dff2.Hour, y=dff2['H_Count'], 
            marker_color ='white',
            #connectgaps=True,
            #stackgroup='one',
        ))
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff2.Hour, y=dff2['Correct'],
            marker_color ='#2E41FE',
            connectgaps=True,
            stackgroup='one',
            marker=dict(size=4, color='#2E41FE'),
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff2.Hour, y=dff2['Incorrect'],
            marker_color ='#F70000',
            connectgaps=True,
            stackgroup='one',
            marker=dict(size=4, color='#F70000'),
        ))
        fig.update_layout(
            title="Number of correct/incorrect answers per hour",
            xaxis_title="Hour",
            yaxis_title="Count",
            height = 350,
            font=dict(size=11),
            hovermode="x unified",
            showlegend = False,
            margin={'t': 50, 'b':80, 'r':50, 'l':80},
            template="plotly_white",
            legend=dict(
                traceorder="reversed",
                title ='User answer',
                font=dict(
                    color="black"
                ),
                bgcolor="white",
                bordercolor="white",
                borderwidth=1
            )
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            showlegend=True,
        )
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    dff2 = pd.DataFrame(columns=[])
    dff2['Count']=dff1['Date'].value_counts()
    dff2['Correct'] = dff1['Date'][dff1['Result'] == 'Correct'].value_counts()
    dff2['Incorrect'] = dff1['Date'][dff1['Result'] == 'Incorrect'].value_counts()
    dff2['Date']=dff1['Date'].value_counts().index
    if input2 == 'Date':
        dff2 = dff2.sort_values('Date', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Total",
            mode="markers+text", x=dff2.Date, y=dff2['Count'], 
            marker_color ='white',
            #connectgaps=True,
            #stackgroup='one',
        ))
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff2.Date, y=dff2['Correct'],
            marker=dict(size=4, color='#2E41FE'),
            connectgaps=True,
            stackgroup='one',
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff2.Date, y=dff2['Incorrect'],
            marker=dict(size=4, color='#F70000'),
            connectgaps=True,
            stackgroup='one',
        ))
        fig.update_layout(
            title="Number of correct/incorrect answers per date",
            xaxis_title="Date",
            yaxis_title="Count",
            height = 350,
            font=dict(size=11),
            hovermode="x unified",
            showlegend = False,
            margin={'t': 50, 'b':80, 'r':50, 'l':80},
            xaxis=dict(tickfont_size=8,rangeslider_visible=True,title= "Date"),
            template="plotly_white",
            legend=dict(
                traceorder="reversed",
                title ='User answer',
                font=dict(
                    color="black"
                ),
                bgcolor="white",
                bordercolor="white",
                borderwidth=1
            )
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            showlegend=True,
        )
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig

    dff2 = pd.DataFrame(columns=[])
    dff2['Count']=dff1['Subject number'].value_counts()
    dff2['Correct'] = dff1['Subject number'][dff1['Result'] == 'Correct'].value_counts()
    dff2['Incorrect'] = dff1['Subject number'][dff1['Result'] == 'Incorrect'].value_counts()
    dff2['Subject']=dff1['Subject number'].value_counts().index
    if input2 == 'Subject number':
        dff2 = dff2.sort_values('Subject', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Total",
            mode="markers+text", x=dff2.Subject, y=dff2['Count'], 
            marker_color ='white',
            #connectgaps=True,
            #stackgroup='one',
        ))
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff2.Subject, y=dff2['Correct'],
            marker_color ='#2E41FE',
            connectgaps=True,
            stackgroup='one',
            marker=dict(size=4, color='#2E41FE'),
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff2.Subject, y=dff2['Incorrect'],
            marker_color ='#F70000',
            connectgaps=True,
            stackgroup='one',
            marker=dict(size=4, color='#F70000'),
        ))
        fig.update_layout(
            title="Number of correct/incorrect answers for different subjects",
            xaxis_title="Subject number",
            yaxis_title="Count",
            height = 350,
            font=dict(size=11),
            hovermode="x unified",
            showlegend = False,
            margin={'t': 50, 'b':80, 'r':50, 'l':80},
            template="plotly_white",
            legend=dict(
                traceorder="reversed",
                title ='User answer',
                font=dict(
                    color="black"
                ),
                bgcolor="white",
                bordercolor="white",
                borderwidth=1
            )
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            showlegend=True,
        )
        fig.layout.plot_bgcolor  = 'white'
        fig.layout.paper_bgcolor = 'white'
        return fig
# ================ call back =====================
@app.callback(
    Output('operator-Student', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_N_Student(input1, reset_click, click_Data):
    return LEDDisplay1(input1, reset_click, click_Data, 'Student ID')

@app.callback(
    Output('operator-A1', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_N_A1(input1, reset_click, click_Data):
    return LEDDisplay2(input1, reset_click, click_Data, 'Correct answer', 'User answer')

@app.callback(
    Output('operator-A2', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_N_A2(input1, reset_click, click_Data):
    return LEDDisplay3(input1, reset_click, click_Data, 'Correct answer', 'User answer')

@app.callback(
    Output('operator-A3', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_N_A3(input1, reset_click, click_Data):
    return LEDDisplay4(input1, reset_click, click_Data, 'Correct answer', 'User answer')

@app.callback(
    Output('operator-A4', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_N_A4(input1, reset_click, click_Data):
    return LEDDisplay5(input1, reset_click, click_Data, 'Correct answer', 'User answer')

@app.callback(
    Output('operator-Correct', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_min_Duration(input1, reset_click, click_Data):
    return LEDDisplay6(input1, reset_click, click_Data, 'Answer duration')

@app.callback(
    Output('operator-Incorrect', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_max_Duration(input1, reset_click, click_Data):
    return LEDDisplay7(input1, reset_click, click_Data, 'Answer duration')

@app.callback(
    Output('operator_N_Question', 'value'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_N_Question(input1, reset_click, click_Data):
    return LEDDisplay8(input1, reset_click, click_Data, 'Question ID')

@app.callback(
    Output('similarity_Q_Categoriy', 'figure'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_Q_Categoriy_plot(input1, reset_click, click_Data):
    return Q_Categoriy_plot(input1, reset_click, click_Data)    

@app.callback(
     Output('opt5', 'options'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')]
)

def update_date_dropdown5(input1, reset_click, click_Data):
    return opt5_update(input1, reset_click, click_Data, 'Student ID')
    
@app.callback(
     Output('opt6', 'options'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')]
)

def update_date_dropdown6(input1, reset_click, click_Data):
    return opt6_update(input1, reset_click, click_Data, 'Student ID')
            
@app.callback(
    Output('Parallel_plot', 'figure'),
    [Input("opt7", "value"), Input('opt5', 'value'), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_Parallel_plot(input1, input2, reset_click, click_Data):
    return similarity_parallel(input1, input2, reset_click, click_Data, 'Class')

@app.callback(
    Output('Parallel_plot1', 'figure'),
    [Input("opt7", "value"), Input('opt6', 'value'), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_Parallel_plot1(input1, input2, reset_click, click_Data):
    return similarity_parallel1(input1, input2, reset_click, click_Data, 'Class')

@app.callback(
    Output('userAnswer-bar', 'figure'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_userAnswer_bar(input1, reset_click, click_Data):
    return bar1(input1, reset_click, click_Data)  

@app.callback(
    Output('heatmap_plot', 'figure'),
    [Input("opt7", "value"), Input('items1', 'value'), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_heatmap_plot(input1, input2, reset_click, click_Data):
    return heatmap_Similarity(input1, input2, reset_click, click_Data)  
          

@app.callback(
    Output('table1', 'children'),
    [Input("opt7", "value"), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_table(input1, reset_click, click_Data):
    return build_table(input1, reset_click, click_Data)

@app.callback(
    Output('visualizations_plot', 'figure'),
    [Input("opt7", "value"), Input('items', 'value'), Input('reset-button', 'n_clicks'), Input('similarity_plot', 'clickData')])

def update_visualizations_plot(input1, input2, reset_click, click_Data):
    return visualizations_plot(input1, input2, reset_click, click_Data)


# ========================================
app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
        generate_learn_button(),
    ],
)

@app.callback(
    [Output("app-content", "children")],
    [Input("app-tabs", "value")],
)

# ======= render the tab contents =======
def render_tab_content(tab_switch):
    return (
        html.Div(
            id="status-container",
            children=[
                build_left_panel(),
                html.Div(
                    id="graphs-container",
                    children=[build_LED(), build_similarity_panel(), build_chart_menu()],
                ),
            ],
        ),
    )

# ======= Callbacks for modal popup =======
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "learn-more-button":
            return {"display": "block"}

    return {"display": "none"}

# Running the server
if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port='8050', debug=True)