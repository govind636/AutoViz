######################### AutoViz New with HoloViews ############################
import numpy as np
import pandas as pd






############# Import from autoviz.AutoViz_Class the following libraries #######
from autoviz.AutoViz_Utils import *
##############   make sure you use: conda install -c pyviz hvplot ###############
import hvplot.pandas  # noqa
import copy
import logging
logging.getLogger("param").setLevel(logging.ERROR)
from bokeh.util.warnings import BokehUserWarning 
import warnings 
warnings.simplefilter(action='ignore', category=BokehUserWarning)
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib import io
import io
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re
import matplotlib
matplotlib.style.use('fivethirtyeight')
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
import os
##########################################################################################
######## This is where we import HoloViews related libraries  #########
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
import panel.widgets as pnw
# import holoviews.plotting.bokeh
######################################################################################
def append_panels(hv_panel, imgdata_list, chart_format):
    imgdata_list.append(hv.output(hv_panel, backend='bokeh', fig=chart_format))
    return imgdata_list

##############################  This is the beginning of the new AutoViz_Holo ###################
def AutoViz_Holo(filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                        lowess=False,chart_format='svg',max_rows_analyzed=150000,
                            max_cols_analyzed=30, save_plot_dir=None):
    """
    ##############################################################################
    ##### AUTOVIZ_HOLO PERFORMS AUTO VISUALIZATION OF ANY DATA SET USING BOKEH. ##
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
    ####  If chart_format='bokeh': Bokeh charts are plotted on Jupyter Notebooks##
    ####  This is the default for AutoViz_Holo.                              #####
    ####  If chart_format='server', dashboards will pop up for each kind of    ###
    ####  chart on your browser.                                             #####
    ####  In both cases, all charts are interactive and you can play with them####
    ####  In the next version, I will let you save them in HTML.             #####
    ##############################################################################
    """
    ####################################################################################
    corr_limit = 0.7  ### This is needed to remove variables correlated above this limit
    ######### create a directory to save all plots generated by autoviz ############
    ############    THis is where you save the figures in a target directory #######
    target_dir = 'AutoViz'
    if not depVar is None:
        if depVar != '':
            target_dir = copy.deepcopy(depVar)
    if save_plot_dir is None:
        mk_dir = os.path.join(".","AutoViz_Plots")
    else:
        mk_dir = copy.deepcopy(save_plot_dir)
    if chart_format == 'html' and not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
        # mk_dir = os.path.join(mk_dir,target_dir)
    if chart_format == 'html' and not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
    ############   Start the clock here and classify variables in data set first ########
    start_time = time.time()
    
    try:
       
        dfin, dep,IDcols,bool_vars,cats,nums,discrete_string_vars,date_vars,classes,problem_type,selected_cols = classify_print_vars(
                                            filename,sep,max_rows_analyzed, max_cols_analyzed,
                                            depVar,dfte,header,verbose)
    except:
        print('Not able to read or load file. Please check your inputs and try again...')
        return None
    #################   This is where the differentiated HoloViews code begins ####
    ls_objects = []
    imgdata_list = list()
    height_size = 400
    width_size = 500
    
    ##########    Now start drawing the Bokeh Plots ###############################
    if len(nums) > 0:
        if problem_type == 'Clustering':
            #### There is no need to do a scatter plot with a dep variable when no dependent variable is given
            print('No scatter plots with depVar when no depVar is given.')
        else:
            drawobj1 = draw_scatters_hv(dfin,nums,chart_format,problem_type,
                          dep, classes, lowess, mk_dir, verbose)
            ls_objects.append(drawobj1)
        ### You can draw pair scatters only if there are 2 or more numeric variables ####
        if len(nums) >= 2:
            drawobj2 = draw_pair_scatters_hv(dfin, nums, problem_type, chart_format, dep,
                           classes, lowess, mk_dir, verbose)
            ls_objects.append(drawobj2)
        ### code comment
    if len(nums) > 0:
        drawobj6 = draw_heatmap_hv(dfin, nums, chart_format, date_vars, dep, problem_type, classes, 
                            mk_dir, verbose)
        ls_objects.append(drawobj6)
    if len(date_vars) > 0:
        drawobj7 = draw_date_vars_hv(dfin,dep,date_vars, nums, chart_format, problem_type, mk_dir, verbose)
        ls_objects.append(drawobj7)
    if len(nums) > 0 and len(cats) > 0:
        drawobj8 = draw_cat_vars_hv(dfin, dep, nums, cats, chart_format, problem_type, mk_dir, verbose)
    print('Time to run AutoViz (in seconds) = %0.0f' %(time.time()-start_time))
    return dfin
####################################################################################
def draw_cat_vars_hv(dfin, dep, nums, cats, chart_format, problem_type, mk_dir, verbose=0):
    ######## SCATTER PLOTS ARE USEFUL FOR COMPARING NUMERIC VARIABLES
    ##### we are going to modify dfin and classes, so we are making copies to make changes
    dft = copy.deepcopy(dfin)
    image_count = 0
    imgdata_list = list()
    N = len(nums)
    cols = 2
    width_size = 600
    height_size = 400
    jitter = 0.05
    alpha = 0.5
    size = 5
    transparent = 0.5
    colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'cat_var_plots'
    #####################################################
    if problem_type == 'Clustering':
        ### There is no depVar in clustering, so no need to add it to None
        pass
    elif problem_type == 'Regression':
        if isinstance(dep, str):
            if dep not in nums:
                nums.append(dep)
        else:
            nums += dep
            nums = find_remove_duplicates(nums)
    else:
        if isinstance(dep, str):
            if dep not in cats:
                cats.append(dep)
        else:
            cats += dep
            cats = find_remove_duplicates(cats)
    ### This is where you draw the widgets #####################
    quantileable = [x for x in nums if len(dft[x].unique()) > 20]
    if len(quantileable) <= 1:
        quantileable = [x for x in nums if len(dft[x].unique()) > 2]
    cmap_list = ['Blues','rainbow', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
    ### The X axis should be cat vars and the Y axis should be numeric vars ######

    



# Save the Panel layout as a responsive HTML file
    x1 = pnw.Select(name='X-Axis', value=cats[0], options=cats, sizing_mode='fixed', width=150)
    y1 = pnw.Select(name='Y-Axis', value=quantileable[0], options=quantileable, sizing_mode='fixed', width=150)

    # you need to decorate this function with depends to make the widgets change axes real time ##
    @pn.depends(x1.param.value, y1.param.value) 
    def create_figure(x1, y1):
        opts = dict(cmap=cmap_list[0], line_color='black')
        #opts['size'] = bubble_size
        opts['alpha'] = alpha
        opts['tools'] = ['hover']
        opts['toolbar'] = 'above'
        opts['colorbar'] = True
        conti_df = dft[[x1,y1]].groupby(x1).mean().reset_index()
        return hv.Bars(conti_df).opts(width=width_size, height=height_size, 
                xrotation=55, title=f"{y1} by {x1}")

    x1.sizing_mode = 'stretch_width'
    y1.sizing_mode = 'stretch_width'
    widgets = pn.WidgetBox(x1, y1, css_classes=['custom-panel-css'])
    
    layout = pn.Column(
        widgets,
        # pn.Row(pn.Spacer(), widgets),
        pn.pane.HoloViews(create_figure, sizing_mode='stretch_both'),  # Wrap the graph in a responsive pane
    #     sizing_mode='stretch_both'  # Make the entire layout responsive
    )
##### code remove 1
    hv_panel=layout

    if chart_format == 'html':
        save_html_data(layout, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)  ### This will display it in a Jupyter Notebook. If you want it on a server, you use drawobj.show()        
    return hv_panel
##########################################################################################
def draw_scatters_hv(dfin, nums, chart_format, problem_type,
                  dep=None, classes=None, lowess=False, mk_dir='AutoViz_Plots', verbose=0):
    ######## SCATTER PLOTS ARE USEFUL FOR COMPARING NUMERIC VARIABLES
    ##### we are going to modify dfin and classes, so we are making copies to make changes
    dfin = copy.deepcopy(dfin)
    dft = copy.deepcopy(dfin)
    image_count = 0
    imgdata_list = list()
    classes = copy.deepcopy(classes)
    N = len(nums)
    cols = 2
    width_size = 600
    height_size = 400
    jitter = 0.05
    alpha = 0.5
    bubble_size = 10
    colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'scatterplots'
    #####################################################
    if problem_type == 'Regression':
        ####### This is a Regression Problem #### You need to plot a Scatter plot ####
        ####### First, Plot each Continuous variable against the Target Variable ###
        ######   This is a Very Simple Way to build an Scatter Chart with One Variable as a Select Variable #######
        alpha = 0.5
        colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
        def load_symbol(symbol, **kwargs):
            color = next(colors)
            return hv.Scatter((dft[symbol].values,dft[dep].values)).opts(framewise=True).opts(size=bubble_size,
                    color=color, alpha=alpha).opts(
                    xlabel='%s' %symbol).opts(ylabel='%s' %dep).opts(
                   title='Scatter Plot of %s against %s variable' %(symbol,dep),responsive=True)
        ### This is where you create the dynamic map and pass it the variable to load the chart!
        dmap = hv.DynamicMap(load_symbol, kdims='Select_Variable').redim.values(Select_Variable=nums).opts(framewise=True)
        ###########  This is where you put the Panel Together ############
        hv_panel = pn.panel(dmap,widget_location='top')
        widgets = hv_panel[1]
        hv_all = pn.Column(pn.Row(*widgets))
        
        if verbose == 2:
            imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
            image_count += 1
    else:
        ##################################################################################################################
        ####### This is a Classification Problem #### You need to plot a Scatter plot ####
        #######   This widget based code works well except it does not add jitter. But it changes y-axis so that's good!
        ##################################################################################################################
        target_vars = dft[dep].unique()
        x = pn.widgets.Select(name='x', options=nums)
        y = pn.widgets.Select(name='y', options=nums)
        kind = pn.widgets.Select(name='kind', value='scatter', options=['scatter'])
        #######  This is where you call the widget and pass it the hv_plotto draw a Chart #######
        hv_plot = dft.hvplot(x=dep, y=y, kind=kind,  size=bubble_size,
                        title='Scatter Plot of each independent numeric variable against target variable',responsive=True)
        # hv_panel = pn.panel(hv_plot)
        hv_all = pn.Column(
                 pn.WidgetBox(y),
                hv_plot
                )
    if chart_format == 'html':
        save_html_data(hv_all, chart_format, plot_name, mk_dir)
    else:
        display(hv_all) 
    
        ### This will display it in a Jupyter Notebook. If you want it on a server, you use drawobj.show()        
    return hv_all
#######################################################################################
def draw_pair_scatters_hv(dfin,nums,problem_type,chart_format, dep=None,
                       classes=None, lowess=False, mk_dir='AutoViz_Plots', verbose=0):
    """
    #### PAIR SCATTER PLOTS ARE NEEDED ONLY FOR CLASSIFICATION PROBLEMS IN NUMERIC VARIABLES
    ### This is where you plot a pair-wise scatter plot of Independent Variables against each other####
    """

    dft = dfin[:]
    image_count = 0
    imgdata_list = list()
    if len(nums) <= 1:
        return
    classes = copy.deepcopy(classes)
    height_size = 400
    width_size = 600
    alpha = 0.5
    bubble_size = 10
    cmap_list = ['rainbow', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
    plot_name = 'pair_scatters'
    ###########################################################################
    if problem_type in ['Regression', 'Clustering']:
        ########## This is for Regression problems ##########
        #########  Here we plot a pair-wise scatter plot of Independent Variables ####
        ### Only 1 color is needed since only 2 vars are plotted against each other ##
        ################################################################################################
        #####  This widgetbox code works but it doesn't change the x- and y-axis when you change variables
        ################################################################################################
        #x = pn.widgets.Select(name='x', options=nums)
        #y = pn.widgets.Select(name='y', options=nums)
        #kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])
        ### Let us say you want to change the range of the x axis 
        ## - Then you can explicitly set the limits and it works!
        #xlimi = (dft[x.value].min(), dft[x.value].max())
        #ylimi = (dft[y.value].min(), dft[y.value].max())
        #plot = dft.hvplot(x=x, y=y, kind=kind,  color=next(colors), alpha=0.5, xlim=xlimi, ylim=ylimi,
        #            title='Pair-wise Scatter Plot of two Independent Numeric variables')
        #hv_panel = pn.Row(pn.WidgetBox(x, y, kind),plot)
        ########################   This is the new way of drawing scatter   ###############################
        
        quantileable = [x for x in nums if len(dft[x].unique()) > 20]
        if len(quantileable) <= 1:
            quantileable = [x for x in nums if len(dft[x].unique()) > 2]
        
##### code remove 2

        
        x1 = pnw.Select(name='X-Axis', value=quantileable[0], options=quantileable, sizing_mode='fixed', width=150)
        y1 = pnw.Select(name='Y-Axis', value=quantileable[1], options=quantileable, sizing_mode='fixed', width=150)
        size = pnw.Select(name='Size', value='None', options=['None'] + quantileable)
        if problem_type == 'Clustering':
            ### There is no depVar in clustering, so no need to add it to None
            color1 = pnw.Select(name='Color', value='None', options=['None'], sizing_mode='fixed', width=150)
        else:
            color1 = pnw.Select(name='Color', value='None', options=['None', dep], sizing_mode='fixed', width=150)
        ## you need to decorate this function with depends to make the widgets change axes real time ##
        @pn.depends(x1.param.value, y1.param.value, color1.param.value) 
        def create_figure(x1, y1, color1):
            opts = dict(cmap=cmap_list[0], width=width_size, height=height_size, line_color='black')
            if color1 != 'None':
                opts['color'] = color1 
            opts['size'] = bubble_size
            opts['alpha'] = alpha
            opts['tools'] = ['hover']
            opts['toolbar'] = 'above'
            opts['colorbar'] = True
            # return hv.Points(dft, [x1, y1], label="%s vs %s" % (x1.title(), y1.title()),
            #     title='Pair-wise Scatter Plot of two Independent Numeric variables').opts(**opts)
            scatter = hv.Points(dft, [x1, y1], label="%s vs %s" % (x1.title(), y1.title()))
            scatter.opts(title='Pair-wise Scatter Plot of two Independent Numeric variables', **opts)
            return scatter
            
        x1.sizing_mode = 'stretch_width'
        y1.sizing_mode = 'stretch_width'
        color1.sizing_mode = 'stretch_width'
        widgets = pn.WidgetBox(x1, y1,color1, css_classes=['custom-panel-css'])
    
        layout = pn.Column(
            widgets,
            pn.pane.HoloViews(create_figure, sizing_mode='stretch_both'),  # Wrap the graph in a responsive pane
    #     sizing_mode='stretch_both'  # Make the entire layout responsive
        )
        hv_panel=layout
        
    else:
        ########## This is for Classification problems ##########
        #########  This is the new way to plot a pair-wise scatter plot ####
        quantileable = [x for x in nums if len(dft[x].unique()) > 20]
        if len(quantileable) <= 1:
            quantileable = [x for x in nums if len(dft[x].unique()) > 2]

##### code remove 3 

        
        x1 = pnw.Select(name='X-Axis', value=quantileable[0], options=quantileable, sizing_mode='fixed', width=150)
        y1 = pnw.Select(name='Y-Axis', value=quantileable[1], options=quantileable, sizing_mode='fixed', width=150)
        size = pnw.Select(name='Size', value='None', options=['None'] + quantileable)
        color1 = pnw.Select(name='Color', value='None', options=['None',dep], sizing_mode='fixed', width=150)
        
        @pn.depends(x1.param.value, y1.param.value, color1.param.value) 
        def create_figure(x1, y1, color1):
            opts = dict(cmap=cmap_list[0], width=width_size, height=height_size, line_color='black')
            if color1 != 'None':
                opts['color'] = color1 
            opts['size'] = bubble_size
            opts['alpha'] = alpha
            opts['tools'] = ['hover']
            opts['toolbar'] = 'above'
            opts['colorbar'] = True
            # return hv.Points(dft, [x1, y1], label="%s vs %s" % (x1.title(), y1.title()),
            #     title='Pair-wise Scatter Plot of two Independent Numeric variables').opts(**opts)
            scatter = hv.Points(dft, [x, y], label="%s vs %s" % (x.title(), y.title()))
            scatter.opts(title='Pair-wise Scatter Plot of two Independent Numeric variables', **opts)
            return scatter
            
        x1.sizing_mode = 'stretch_width'
        y1.sizing_mode = 'stretch_width'
        color1.sizing_mode = 'stretch_width'
        widgets = pn.WidgetBox(x1, y1,color1, css_classes=['custom-panel-css'])
    
        layout = pn.Column(
            widgets,
            pn.pane.HoloViews(create_figure, sizing_mode='stretch_both'),  # Wrap the graph in a responsive pane
    #     sizing_mode='stretch_both'  # Make the entire layout responsive
        )

        hv_panel=layout
       
    if chart_format == 'html':
        save_html_data(layout, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)  ### This will display it in a Jupyter Notebook. If you want it on a server, you use drawobj.show()   
    
    return hv_panel

###########################################################################
def draw_date_vars_hv(df,dep,datevars, nums, chart_format, modeltype='Regression',
                        mk_dir='AutoViz_Plots', verbose=0):
    #### Now you want to display 2 variables at a time to see how they change over time
    ### Don't change the number of cols since you will have to change rows formula as well
    df = copy.deepcopy(df)
    imgdata_list = list()
    image_count = 0
    N = len(nums)
    cols = 2
    width_size = 600
    height_size = 400
    jitter = 0.05
    alpha = 0.5
    size = 5
    transparent = 0.5
    colortext = 'brycgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    plot_name = 'timeseries_plots'
    #####################################################
    ###### Draw the time series for Regression and DepVar
    #####################################################
    if modeltype == 'Regression':
        if isinstance(dep, str):
            if dep not in nums:
                nums.append(dep)
        else:
            nums += dep
            nums = find_remove_duplicates(nums)
    else:
            nums = find_remove_duplicates(nums)
    ### This is where you draw the widgets #####################
    quantileable = nums[:]
    cmap_list = ['Blues','rainbow', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

#### code remove 4  

    
    #####################################################
    x1 = pnw.Select(name='X-Axis', value=datevars[0], options=datevars, sizing_mode='fixed', width=150)
    y1 = pnw.Select(name='Y-Axis', value=quantileable[0], options=quantileable, sizing_mode='fixed', width=150)

    ## you need to decorate this function with depends to make the widgets change axes real time ##
    @pn.depends(x1.param.value, y1.param.value) 
    def create_figure1(x1, y1):
        opts = dict(cmap=cmap_list[0], line_color='black')
        #opts['size'] = bubble_size
        opts['alpha'] = alpha
        opts['tools'] = ['hover']
        opts['toolbar'] = 'above'
        opts['colorbar'] = True
        dft = df.set_index(df[x])
        conti_df = df[[x1,y1]].set_index(df[x1]).drop(x1, axis=1)
        return hv.Curve(conti_df).opts(
            line_width=1, line_color=next(colors),line_dash='dotted', line_alpha=0.5).opts(
            width=width_size, height=height_size,title='Time Series plots of Numeric vars')

 
    x1.sizing_mode = 'stretch_width'
    y1.sizing_mode = 'stretch_width'
    widgets = pn.WidgetBox(x1, y1, css_classes=['custom-panel-css'])
    
    layout = pn.Column(
        widgets,
        pn.pane.HoloViews(create_figure1, sizing_mode='stretch_both'),  # Wrap the graph in a responsive pane
    #     sizing_mode='stretch_both'  # Make the entire layout responsive
    )
    hv_panel=layout
                            

    if chart_format == 'html':
        save_html_data(layout, chart_format, plot_name, mk_dir)
    else:
        ### This will display it in a Jupyter Notebook. If you want it on a server, you use drawobj.show()        
        display(hv_panel)  
    return hv_panel
################################################################################################
############# Draw a Heatmap using Pearson Correlation #########################################
################################################################################################
def draw_heatmap_hv(dft, conti, chart_format, datevars=[], dep=None,
                            modeltype='Regression',classes=None, mk_dir='AutoViz_Plots', verbose=0):
    dft = copy.deepcopy(dft)
    ### Test if this is a time series data set, then differene the continuous vars to find
    ###  if they have true correlation to Dependent Var. Otherwise, leave them as is
    width_size = 600
    height_size = 400
    cmap_list = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    if len(conti) <= 1:
        return
    elif len(conti) <= 10:
        height_size = 500
        width_size = 600
    else:
        height_size = 800
        width_size = 1200
    plot_name = 'heatmaps'
    #####  If it is a datetime index we need to calculate heat map on differenced data ###
    if isinstance(dft.index, pd.DatetimeIndex) :
        dft = dft[:]
        timeseries_flag = True
        pass
    else:
        dft = dft[:]
        try:
            dft.index = pd.to_datetime(dft.pop(datevars[0]),infer_datetime_format=True)
            timeseries_flag = True
        except:
            if verbose == 1 and len(datevars) > 0:
                print('No date vars could be found or %s could not be indexed.' %datevars)
            elif verbose == 1 and len(datevars) == 0:
                print('No date vars could be found in data set')
            timeseries_flag = False
    # Add a column: the color depends on target variable but you can use whatever function
    imgdata_list = list()
    ##########    This is where we plot the charts #########################
    if not modeltype in ['Regression','Clustering']:
        ########## This is for Classification problems only ###########
        if dft[dep].dtype == object or dft[dep].dtype == np.int64:
            dft[dep] = dft[dep].factorize()[0]
        image_count = 0
        N = len(conti)
        target_vars = dft[dep].unique()
        plotc = 1
        #rows = len(target_vars)
        rows = 1
        cols = 1
        if timeseries_flag:
            dft_target = dft[[dep]+conti].diff()
        else:
            dft_target = dft[:]
        dft_target[dep] = dft[dep].values
        corre = dft_target.corr()
        if timeseries_flag:
            heatmap = corre.hvplot.heatmap( colorbar=True, 
                    cmap=cmap_list, rot=70,
            title='Time Series: Heatmap of all Differenced Continuous vars for target = %s' %dep,responsive=True)
        else:
            heatmap = corre.hvplot.heatmap( colorbar=True,
                    cmap=cmap_list,
                    rot=70,
            title='Heatmap of all Continuous Variables including target',responsive=True);
        hv_plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_font_size='7pt'))
        hv_panel = pn.panel(hv_plot)
        # if verbose == 2:
        #     imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
        #     image_count += 1
    else:
        ### This is for Regression and None Dep variable problems only ##
        image_count = 0
        if dep is None or dep == '':
            pass
        else:
            conti += [dep]
        dft_target = dft[conti]
        if timeseries_flag:
            dft_target = dft_target.diff().dropna()
        else:
            dft_target = dft_target[:]
        N = len(conti)
        corre = dft_target.corr()
        if timeseries_flag:
            heatmap = corre.hvplot.heatmap( colorbar=True, 
                    cmap=cmap_list,
                                           rot=70,
                title='Time Series Data: Heatmap of Differenced Continuous vars including target',responsive=True).opts(
                        opts.HeatMap(tools=['hover'], toolbar='above'))
        else:
            heatmap = corre.hvplot.heatmap( colorbar=True, 
                    cmap=cmap_list,
                                           rot=70,
            title='Heatmap of all Continuous Variables including target',responsive=True).opts(
                                    opts.HeatMap(tools=['hover'],  toolbar='above'))
        hv_plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_font_size='7pt'))
        hv_panel = pn.panel(hv_plot)
    #     if verbose == 2:
    #         imgdata_list = append_panels(hv_panel, imgdata_list, chart_format)
    #         image_count += 1
    # ############# End of Heat Maps ##############
    # if chart_format in ['server', 'bokeh_server', 'bokeh-server']:
    #     print('%s can be found in URL below:' %plot_name)
    #     server = pn.serve(hv_panel, start=True, show=True)
        #hv_panel.show() ### dont use show for just heatmap there is some problem with it
    if chart_format == 'html':
        save_html_data(hv_panel, chart_format, plot_name, mk_dir)
    else:
        display(hv_panel)  ### This will display it in a Jupyter Notebook. If you want it on a server, you use drawobj.show()        
        #display_obj(hv_panel)  ### This will display it in a Jupyter Notebook. If you want it on a server, you use drawobj.show()
    return hv_panel
#######################################################################################
