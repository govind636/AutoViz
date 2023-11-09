############################################################################
#Copyright 2019 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#################################################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import os
####################################################################################
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib import io
import io
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re
import matplotlib
import seaborn
matplotlib.style.use('seaborn-v0_8')
from itertools import cycle, combinations
from collections import defaultdict
import copy
import time
import sys
import random
import xlrd
from io import BytesIO
import base64
import traceback

##########################################################################################
from autoviz.AutoViz_Holo import AutoViz_Holo

#############################################################################################
class AutoViz_Class():
    """
        ##############################################################################
        #############       This is not an Officially Supported Google Product! ######
        ##############################################################################
        #Copyright 2019 Google LLC                                              ######
        #                                                                       ######
        #Licensed under the Apache License, Version 2.0 (the "License");        ######
        #you may not use this file except in compliance with the License.       ######
        #You may obtain a copy of the License at                                ######
        #                                                                       ######
        #    https://www.apache.org/licenses/LICENSE-2.0                        ######
        #                                                                       ######
        #Unless required by applicable law or agreed to in writing, software    ######
        #distributed under the License is distributed on an "AS IS" BASIS,      ######
        #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.#####
        #See the License for the specific language governing permissions and    ######
        #limitations under the License.                                         ######
        ##############################################################################
        ###########             AutoViz Class                                   ######
        ###########             by Ram Seshadri                                 ######
        ###########      AUTOMATICALLY VISUALIZE ANY DATA SET                   ######
        ###########            Version V0.0.68 1/10/20                          ######
        ##############################################################################
        ##### AUTOVIZ PERFORMS AUTOMATIC VISUALIZATION OF ANY DATA SET WITH ONE CLICK.
        #####    Give it any input file (CSV, txt or json) and AV will visualize it.##
        ##### INPUTS:                                                            #####
        #####    A FILE NAME OR A DATA FRAME AS INPUT.                           #####
        ##### AutoViz will visualize any sized file using a statistically valid sample.
        #####  - COMMA is assumed as default separator in file. But u can change it.##
        #####  - Assumes first row as header in file but you can change it.      #####
        #####  - First instantiate an AutoViz class to  hold output of charts, plots.#
        #####  - Then call the Autoviz program with inputs as defined below.       ###
        ##############################################################################
    """
    def __init__(self):
        self.overall = {
        'name': 'overall',
        'plots': [],
        'heading': [],
        'subheading':[],  #"\n".join(subheading)
        'desc': [],  #"\n".join(subheading)
        'table1_title': "",
        'table1': [],
        'table2_title': "",
        'table2': []
            }  ### This is for overall description and comments about the data set
        self.scatter_plot = {
        'name': 'scatter',
        'heading': 'Scatter Plot of each Continuous Variable against Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ##### This is for description and images for scatter plots ###
        self.pair_scatter = {
        'name': 'pair-scatter',
        'heading': 'Pairwise Scatter Plot of each Continuous Variable against other Continuous Variables',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': []  #"\n".join(desc)
        }   ##### This is for description and images for pairs of scatter plots ###
        self.dist_plot = {
        'name': 'distribution',
        'heading': 'Distribution Plot of Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': []  #"\n".join(desc)
        } ##### This is for description and images for distribution plots ###
        self.pivot_plot = {
        'name': 'pivot',
        'heading': 'Pivot Plots of all Continuous Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        } ##### This is for description and images for pivot  plots ###
        self.violin_plot = {
        'name': 'violin',
        'heading': 'Violin Plots of all Continuous Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ##### This is for description and images for violin plots ###
        self.heat_map = {
        'name': 'heatmap',
        'heading': 'Heatmap of all Continuous Variables for target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }   ##### This is for description and images for heatmaps ###
        self.bar_plot = {
        'name': 'bar',
        'heading': 'Bar Plots of Average of each Continuous Variable by Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ##### This is for description and images for bar plots ###
        self.date_plot = {
        'name': 'time-series',
        'heading': 'Time Series Plots of Two Continuous Variables against a Date/Time Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ######## This is for description and images for date time plots ###
        self.wordcloud = {
        'name': 'wordcloud',
        'heading': 'Word Cloud Plots of NLP or String vars',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ######## This is for description and images for date time plots ###
        self.catscatter_plot = {
        'name': 'catscatter',
        'heading': 'Cat-Scatter  Plots of categorical vars',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ######## This is for description and images for catscatter plots ###


    def add_plots(self,plotname,X):
        """
        This is a simple program to append the input chart to the right variable named plotname
        which is an attribute of class AV. So make sure that the plotname var matches an exact
        variable name defined in class AV. Otherwise, this will give an error.
        """
        if X is None:
            ### If there is nothing to add, leave it as it is.
            #print("Nothing to add Plot not being added")
            pass
        else:
            getattr(self, plotname)["plots"].append(X)

    def add_subheading(self,plotname,X):
        """
        This is a simple program to append the input chart to the right variable named plotname
        which is an attribute of class AV. So make sure that the plotname var matches an exact
        variable name defined in class AV. Otherwise, this will give an error.
        """
        if X is None:
            ### If there is nothing to add, leave it as it is.
            pass
        else:
            getattr(self,plotname)["subheading"].append(X)

    def AutoViz(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=1,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,
                                max_cols_analyzed=30, save_plot_dir=None):
        """
        ##############################################################################
        ##### AUTOVIZ PERFORMS AUTOMATIC VISUALIZATION OF ANY DATA SET WITH ONE CLICK.
        #####    Give it any input file (CSV, txt or json) and AV will visualize it.##
        ##### INPUTS:                                                            #####
        #####    A FILE NAME OR A DATA FRAME AS INPUT.                           #####
        ##### AutoViz will visualize any sized file using a statistically valid sample.
        #####  - max_rows_analyzed = 150000 ### this limits the max number of rows ###
        #####           that is used to display charts                             ###
        #####  - max_cols_analyzed = 30  ### This limits the number of continuous  ###
        #####           vars that can be analyzed                                 ####
        #####  - COMMA is assumed as default separator in file. But u can change it.##
        #####  - Assumes first row as header in file but you can change it.      #####
        #####  - First instantiate an AutoViz class to  hold output of charts, plots.#
        #####  - Then call the Autoviz program with inputs as defined below.       ###
        ##############################################################################
        ##### This is the main calling program in AV. It will call all the load, #####
        ####  display and save rograms that are currently outside AV. This program ###
        ####  will draw scatter and other plots for the input data set and then   ####
        ####  call the correct variable name with add_plots function and send in  ####
        ####  the chart created by that plotting program, for example, scatter   #####
        ####  You have to make sure that add_plots function has the exact name of ####
        ####  the variable defined in the Class AV. If not, this will give an error.##
        ####  If verbose=0: it does not print any messages and goes into silent mode##
        ####  This is the default.                                               #####
        ####  If verbose=1, it will print messages on the terminal and also display###
        ####  charts on terminal                                                 #####
        ####  If verbose=2, it will print messages but will not display charts,  #####
        ####  it will simply save them.                                          #####
        ##############################################################################
        """
        if isinstance(depVar, list):
            print('Since AutoViz cannot visualize multi-label targets, choosing first item in targets: %s' %depVar[0])
            depVar = depVar[0]
        
        ####################################################################################
        if chart_format.lower() in ['bokeh','server','bokeh_server','bokeh-server', 'html']:
            dft = AutoViz_Holo(filename, sep, depVar, dfte, header, verbose,
                        lowess,chart_format,max_rows_analyzed,
                            max_cols_analyzed, save_plot_dir)
        else:
            dft = self.AutoViz_Main(filename, sep, depVar, dfte, header, verbose,
                        lowess,chart_format,max_rows_analyzed,
                            max_cols_analyzed, save_plot_dir)
        return dft
    
   
