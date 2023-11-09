
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
# ####################################################################################
import matplotlib
matplotlib.use('agg')
import io
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re
import matplotlib
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
######## This is where we import HoloViews related libraries  #########
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
import panel.widgets as pnw
from .classify_method import classify_columns
######## This is where we store the image data in a dictionary with a list of images #########

#### This module analyzes a dependent Variable and finds out whether it is a
#### Regression or Classification type problem

def save_html_data(hv_all, chart_format, plot_name, mk_dir, additional=''):

    print('Saving %s in HTML format' %(plot_name+additional))
    if not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
    if additional == '':
        filename = os.path.join(mk_dir,plot_name+"."+chart_format)
    else:
        filename = os.path.join(mk_dir,plot_name+additional+"."+chart_format)

        pn.panel(hv_all).save(filename, embed=True)
    

def analyze_problem_type(train, target, verbose=0) : 
    train = copy.deepcopy(train)
    target = copy.deepcopy(target)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    
    if isinstance(target, str):
        target = [target]
    ### we can analyze only the first target in a multi-label to detect problem type ##
    targ = target[0]
    ####  This is where you detect what kind of problem it is #################
    if  train[targ].dtype in [np.int64,np.int32,np.int16,np.int8]:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique().tolist()) > 2 and len(train[targ].unique().tolist()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float16','float32','float64']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique().tolist()) > 2 and len(train[targ].unique().tolist()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif train[targ].dtype == bool:
        model_class = 'Binary_Classification'
    else:
        if len(train[targ].unique().tolist()) <= 2:
            model_class = 'Binary_Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose <= 2:
        print('''\n################ %s problem #####################''' %model_class)
    return model_class
#################################################################################
# Pivot Tables are generally meant for Categorical Variables on the axes
# and a Numeric Column (typically the Dep Var) as the "Value" aggregated by Sum.
# Let's do some pivot tables to capture some meaningful insights

#Bar Plots are for 2 Categoricals and One Numeric (usually Dep Var)
def plot_fast_average_num_by_cat(dft, cats, num_vars, verbose=0,kind="bar"):
    """
    Great way to plot continuous variables fast grouped by a categorical variable. Just sent them in and it will take care of the rest!
    """
    chunksize = 20
    stringlimit = 20
    col = 2
    width_size = 15
    height_size = 4
    N = int(len(num_vars)*len(cats))
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    if N % 2 == 0:
        row = N//col
    else:
        row = int(N//col + 1)
    fig = plt.figure()
    if kind == 'bar':
        fig.suptitle('Bar plots for each Continuous by each Categorical variable', fontsize=15,y=1.01)
    else:
        fig.suptitle('Time Series plots for all date-time vars %s' %cats, fontsize=15,y=1.01)
    if col < 2:
        fig.set_size_inches(min(15,8),row*5)
        fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    else:
        fig.set_size_inches(min(col*10,20),row*5)
        fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    counter = 1
    for cat in cats:
        for each_conti in num_vars:
            color3 = next(colors)
            try:
                ax1 = plt.subplot(row, col, counter)
                if kind == "bar":
                    data = dft.groupby(cat)[each_conti].mean().sort_values(
                            ascending=False).head(chunksize)
                    data.plot(kind=kind,ax=ax1,color=color3)
                elif kind == "line":
                    data = dft.groupby(cat)[each_conti].mean().sort_index(
                            ascending=True).head(chunksize)
                    data.plot(kind=kind,ax=ax1,color=color3)
                if dft[cat].dtype == object or str(dft[cat].dtype) in ['category']:
                    labels = data.index.str[:stringlimit].tolist()
                else:
                    labels = data.index.tolist()
                ax1.set_xlabel("")
                ax1.set_xticklabels(labels,fontdict={'fontsize':9}, rotation = 45, ha="right")
                ax1.set_title('Average %s by %s (Top %d)' %(each_conti,cat,chunksize))
                counter += 1
            except:
                ax1.set_title('No plot as %s is not numeric' %each_conti)
                counter += 1
    if verbose <= 1:
        plt.show()
    if verbose == 2:
        return fig




# ######################################################################################
# # This little function classifies columns into 4 types: categorical, continuous, boolean and
# # certain columns that have only one value repeated that they are useless and must be removed from dataset
# #Subtract RIGHT_LIST from LEFT_LIST to produce a new list
# ### This program is USED VERY HEAVILY so be careful about changing it
def list_difference(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

######## Find ANY word in columns to identify ANY TYPE OF columns
####### search_for_list = ["Date","DATE", "date"], any words you want to search for it
####### columns__list and word refer to columns in the dataset that is the target dataset
####### Both columns_list and search_for_list must be lists - otherwise it won't work

############################################## check###########################################################################
def search_for_word_in_list(columns_list, search_for_list):
    columns_list = columns_list[:]
    search_for_list = search_for_list[:]
    lst=[]
    for src in search_for_list:
        for word in columns_list:
            result = re.findall (src, word)
            if len(result)>0:
                if word.endswith(src) and not word in lst:
                    lst.append(word)
            elif (word == 'id' or word == 'ID') and not word in lst:
                lst.append(word)
            else:
                continue
    return lst

### This is a small program to look for keywords such as "id" in a dataset to see if they are ID variables
### If that doesn't work, it then compares the len of the dataframe to the variable's unique values. If
###	they match, it means that the variable could be an ID variable. If not, it goes with the name of
###	of the ID variable through a keyword match with "id" or some such keyword in dataset's columns.
###  This is a small program to look for keywords such as "id" in a dataset to see if they are ID variables
###    If that doesn't work, it then compares the len of the dataframe to the variable's unique values. If
###     they match, it means that the variable could be an ID variable. If not, it goes with the name of
###     of the ID variable through a keyword match with "id" or some such keyword in dataset's columns.

def analyze_ID_columns(dfin,columns_list):
    columns_list = columns_list[:]
    dfin = dfin[:]
    IDcols_final = []
    IDcols = search_for_word_in_list(columns_list,
        ['ID','Identifier','NUMBER','No','Id','Num','num','_no','.no','Number','number','_id','.id'])
    if IDcols == []:
        for eachcol in columns_list:
            if len(dfin) == len(dfin[eachcol].unique()) and dfin[eachcol].dtype != float:
                IDcols_final.append(eachcol)
    else:
        for each_col in IDcols:
            if len(dfin) == len(dfin[each_col].unique()) and dfin[each_col].dtype != float:
                IDcols_final.append(each_col)
    if IDcols_final == [] and IDcols != []:
        IDcols_final = IDcols
    return IDcols_final


############################################## check###########################################################################

# THESE FUNCTIONS ASSUME A DIRTY DATASET" IN A PANDAS DATAFRAME AS Inum_j}lotsUT
# AND CONVERT THEM INTO A DATASET FIT FOR ANALYSIS IN THE END
# In [ ]:
# this function starts with dividing columns into 4 types: categorical, continuous, boolean and to_delete
# The To_Delete columns have only one unique value and can be removed from the dataset
def start_classifying_vars(dfin, verbose):
    dfin = dfin[:]
    cols_to_delete = []
    boolean_vars = []
    categorical_vars = []
    continuous_vars = []
    discrete_vars = []
    totrows = dfin.shape[0]
    if totrows == 0:
        print('Error: No rows in dataset. Check your input again...')
        return cols_to_delete, boolean_vars, categorical_vars, continuous_vars, discrete_vars, dfin
    for col in dfin.columns:
        if col == 'source':
            continue
        elif len(dfin[col].value_counts()) <= 1:
            cols_to_delete.append(dfin[col].name)
            print('    Column %s has only one value hence it will be dropped' %dfin[col].name)
        elif dfin[col].dtype==object:
            if (dfin[col].str.len()).any()>50:
                cols_to_delete.append(dfin[col].name)
                continue
            elif search_for_word_in_list([col],['DESCRIPTION','DESC','desc','Text','text']):
                cols_to_delete.append(dfin[col].name)
                continue
            elif len(dfin.groupby(col)) == 1:
                cols_to_delete.append(dfin[col].name)
                continue
            elif dfin[col].isnull().sum() > 0:
                missing_rows=dfin[col].isnull().sum()
                pct_missing = float(missing_rows)/float(totrows)
                if pct_missing > 0.90:
                    if verbose <= 1:
                        print('Pct of Missing Values in %s exceed 90 pct, hence will be dropped...' %col)
                    cols_to_delete.append(dfin[col].name)
                    continue
                elif len(dfin.groupby(col)) == 2:
                    boolean_vars.append(dfin[col].name)
                    py_version = sys.version_info[0]
                    if py_version < 3:
                        # This is the Python 2 Version
                        try:

                            item_mode = dfin[col].mode().mode[0]
                        except:
                            print('''Scipy.stats package not installed in your Python2. Get it installed''')
                    else:
                        # This is the Python 3 Version
                        try:

                            item_mode = dfin[col].mode()[0]
                        except:
                            print('''Statistics package not installed in your Python3. Get it installed''')
                    ### Remember that fillna only works at dataframe level! ###
                    dfin[[col]] = dfin[[col]].fillna(item_mode)
                    continue
                elif len(dfin.groupby(col)) < 20 and len(dfin.groupby(col)) > 1:
                    categorical_vars.append(dfin[col].name)
                    continue
                else:
                    discrete_vars.append(dfin[col].name)
                    continue
            elif len(dfin.groupby(col)) == 2:
                boolean_vars.append(dfin[col].name)
                continue
            elif len(dfin.groupby(col)) < 20 and len(dfin.groupby(col)) > 1:
                categorical_vars.append(dfin[col].name)
                continue
            else:
                discrete_vars.append(dfin[col].name)
        elif dfin[col].dtype=='int64' or dfin[col].dtype=='int32':
            if len(dfin[col].value_counts()) <= 15:
                categorical_vars.append(dfin[col].name)
        else:
            if dfin[col].isnull().sum() > 0:
                missing_rows=dfin[col].isnull().sum()
                pct_missing = float(missing_rows)/float(totrows)
                if pct_missing > 0.90:
                    if verbose <= 1:
                        print('Pct of Missing Values in %s exceed 90 pct, hence will be dropped...' %col)
                    cols_to_delete.append(dfin[col].name)
                    continue
                elif len(dfin.groupby(col)) == 2:
                    boolean_vars.append(dfin[col].name)
                    py_version = sys.version_info[0]
                    if py_version < 3:
                        # This is the Python 2 Version
                        try:

                            item_mode = dfin[col].mode().mode[0]
                        except:
                            print('''Scipy.stats package not installed in your Python2. Get it installed''')
                    else:
                        # This is the Python 3 Version
                        try:

                            item_mode = dfin[col].mode()[0]
                        except:
                            print('''Statistics package not installed in your Python3. Get it installed''')
                    ### Remember that fillna only works at dataframe level! ###
                    dfin[[col]] = dfin[[col]].fillna(item_mode)
                    continue
                else:
                    if len(dfin[col].value_counts()) <= 25 and len(dfin) >= 250:
                        categorical_vars.append(dfin[col].name)
                    else:
                        continuous_vars.append(dfin[col].name)
            elif len(dfin.groupby(col)) == 2:
                boolean_vars.append(dfin[col].name)
                continue
            else:
                if len(dfin[col].value_counts()) <= 25 and len(dfin) >= 250:
                    categorical_vars.append(dfin[col].name)
                else:
                    continuous_vars.append(dfin[col].name)
    return cols_to_delete, boolean_vars, categorical_vars, continuous_vars, discrete_vars, dfin

#### this is the MAIN ANALYSIS function that calls the start_classifying_vars and then
#### takes that result and divides categorical vars into 2 additional types: discrete vars and bool vars
def analyze_columns_in_dataset(dfx,IDcolse,verbose):
    dfx = dfx[:]
    IDcolse = IDcolse[:]
    cols_delete, bool_vars, cats, nums, discrete_string_vars, dft = start_classifying_vars(dfx,verbose)
    continuous_vars = nums
    if nums != []:
        for k in nums:
            if len(dft[k].unique())==2:
                bool_vars.append(k)
            elif len(dft[k].unique())<=20:
                cats.append(k)
            elif (np.array(dft[k]).dtype=='float64' or np.array(dft[k]).dtype=='int64') and (k not in continuous_vars):
                if len(dft[k].value_counts()) <= 25:
                    cats.append(k)
                else:
                    continuous_vars.append(k)
            elif dft[k].dtype==object:
                discrete_string_vars.append(k)
            elif k in continuous_vars:
                continue
            else:
                print('The %s variable could not be classified into any known type' % k)
    #print(cols_delete, bool_vars, cats, continuous_vars, discrete_string_vars)
    date_vars = search_for_word_in_list(dfx.columns.tolist(),['Date','DATE','date','TIME','time',
                                                   'Time','Year','Yr','year','yr','timestamp',
                                                   'TimeStamp','TIMESTAMP','Timestamp','Time Stamp'])
    date_vars = [x for x in date_vars if x not in find_remove_duplicates(cats+bool_vars) ]
    if date_vars == []:
        for col in continuous_vars:
            if dfx[col].dtype==int:
                if dfx[col].min() > 1900 or dfx[col].max() < 2100:
                    date_vars.append(col)
        for col in discrete_string_vars:
            try:
                dfx.index = pd.to_datetime(dfx.pop(col), infer_datetime_format=True)
            except:
                continue
    if isinstance(dfx.index, pd.DatetimeIndex):
        date_vars = [dfx.index.name]
    continuous_vars=list_difference(list_difference(continuous_vars,date_vars),IDcolse)
    #cats =  list_difference(continuous_vars, cats)
    cats=list_difference(cats,date_vars)
    discrete_string_vars=list_difference(list_difference(discrete_string_vars,date_vars),IDcolse)
    return cols_delete, bool_vars, cats, continuous_vars, discrete_string_vars,date_vars, dft

# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
#################################################################################
def load_file_dataframe(dataname, sep=",", header=0, verbose=0, nrows=None,parse_dates=False):

    start_time = time.time()
    ###########################  This is where we load file or data frame ###############
    
    if isinstance(dataname,str):
        #### this means they have given file name as a string to load the file #####
        codex_flag = False
        codex = ['ascii', 'utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        if dataname != '' and dataname.endswith(('csv')):
            try:
                dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None,
                                parse_dates=parse_dates)
                if not nrows is None:
                    if nrows < dfte.shape[0]:
                        print('    max_rows_analyzed is smaller than dataset shape %d...' %dfte.shape[0])
                        dfte = dfte.sample(nrows, replace=False, random_state=99)
                        print('        randomly sampled %d rows from read CSV file' %nrows)
                print('Shape of your Data Set loaded: %s' %(dfte.shape,))
                if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
                    print('You have duplicate column names in your data set. Removing duplicate columns now...')
                    dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
                return dfte
            except:
                codex_flag = True
        if codex_flag:
            for code in codex:
                try:
                    dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=code, nrows=nrows,
                                    skiprows=skip_function, parse_dates=parse_dates)
                except:
                    print('    pandas %s encoder does not work for this file. Continuing...' %code)
                    continue
        elif dataname.endswith(('xlsx','xls','txt')):
            #### It's very important to get header rows in Excel since people put headers anywhere in Excel#
            if nrows is None:
                dfte = pd.read_excel(dataname,header=header, parse_dates=parse_dates)
            else:
                dfte = pd.read_excel(dataname,header=header, nrows=nrows, parse_dates=parse_dates)
            print('Shape of your Data Set loaded: %s' %(dfte.shape,))
            return dfte
        else:
            print('    Filename is an empty string or file not able to be loaded')
            return None

        

    elif isinstance(dataname,pd.DataFrame):
        #### this means they have given a dataframe name to use directly in processing #####
        if nrows is None:
            dfte = copy.deepcopy(dataname)
        else:
            if nrows < dataname.shape[0]:
                print('    Since nrows is smaller than dataset, loading random sample of %d rows into pandas...' %nrows)
                dfte = dataname.sample(n=nrows, replace=False, random_state=99)
            else:
                dfte = copy.deepcopy(dataname)
        print('Shape of your Data Set loaded: %s' %(dfte.shape,))
        if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
            print('You have duplicate column names in your data set. Removing duplicate columns now...')
            dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
        return dfte
    else:
        print(type(dataname))
        print('Dataname input must be a filename with path to that file or a Dataframe')
        return None
##########################################################################################
import copy
def classify_print_vars(filename,sep, max_rows_analyzed, max_cols_analyzed,
                        depVar='',dfte=None, header=0,verbose=0):
    corr_limit = 0.7  ### This limit represents correlation above this, vars will be removed
    
    start_time=time.time()
    
    if filename:
        dataname = copy.deepcopy(filename)
        parse_dates = True
    else:
        dataname = copy.deepcopy(dfte)
        parse_dates = True
    
    dfte = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, 
                    nrows=max_rows_analyzed, parse_dates=parse_dates)
    
    
    orig_preds = [x for x in list(dfte) if x not in [depVar]]
    #################    CLASSIFY  COLUMNS   HERE    ######################
    if len(dfte) >= 100000:
        dfte_small = dfte.sample(n=10000, random_state=99)
    else:
        dfte_small = copy.deepcopy(dfte)
    var_df = classify_columns(dfte_small[orig_preds], verbose)
    #####       Classify Columns   ################
    IDcols = var_df['id_vars']
    discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    int_vars = var_df['int_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] + int_vars + bool_vars
    date_vars = var_df['date_vars']
    
    if len(var_df['continuous_vars'])==0 and len(int_vars)>0:
        
        continuous_vars = var_df['int_vars']
        categorical_vars = list_difference(categorical_vars, int_vars)
        
        int_vars = []
        
    #elif len(var_df['continuous_vars'])==0 and len(int_vars)==0:
    #    print('Cannot visualize this dataset since no numeric or integer vars in data...returning')
    #    return dataname
    else:
        
        continuous_vars = var_df['continuous_vars']
    #### from now you can use wordclouds on discrete_string_vars ######################
    preds = [x for x in orig_preds if x not in IDcols+cols_delete]

    if len(IDcols+cols_delete) == 0:
        print('        No variables removed since no ID or low-information variables found in data set')
    else:
        print('        %d variable(s) removed since they were ID or low-information variables'
                                %len(IDcols+cols_delete))
        if verbose >= 1:
            print('        List of variables removed: %s' %(IDcols+cols_delete))
    #############    Sample data if too big and find problem type   #############################
   
    if dfte.shape[0]>= max_rows_analyzed:
        print('Since Number of Rows in data %d exceeds maximum, randomly sampling %d rows for EDA...' %(len(dfte),max_rows_analyzed))
        dft = dfte.sample(max_rows_analyzed, random_state=0)
    else:
        dft = copy.deepcopy(dfte)
    ###### This is where you find what type the dependent variable is ########
    if isinstance(depVar, list):
        # If depVar is a list, just select the first one in the list to visualize!
        depVar = depVar[0]
        print('Since AutoViz cannot visualize multi-label targets, selecting %s from target list' %depVar[0])
    ### Now we analyze depVar as usual - Do not change the next line to elif! ###
    
    if type(depVar) == str:
        if depVar == '':
            cols_list = list(dft)
            problem_type = 'Clustering'
            classes = []
        else:
            try:
                problem_type = analyze_problem_type(dft, depVar,verbose)
            except:
                print('Could not find given target var in data set. Please check input')
                ### return the data frame as is ############
                return dfte
            cols_list = list_difference(list(dft),depVar)
            if dft[depVar].dtype == object:
                classes = dft[depVar].unique().tolist()
                #### You dont have to convert it since most charts can take string vars as target ####
                #dft[depVar] = dft[depVar].factorize()[0]
            elif str(dft[depVar].dtype) in ['category']:
                #### You dont have to convert it since most charts can take string vars as target ####
                classes = dft[depVar].unique().tolist()
            elif dft[depVar].dtype in [np.int64, np.int32, np.int16, np.int8]:
                classes = dft[depVar].unique().tolist()
            elif dft[depVar].dtype == bool:
                dft[depVar] = dft[depVar].astype(int)
                classes =  dft[depVar].unique().astype(int).tolist()
            elif dft[depVar].dtype == float and problem_type.endswith('Classification'):
                classes = dft[depVar].factorize()[1].tolist()
            else:
                classes = dft[depVar].factorize()[1].tolist()
    elif depVar == None:
            cols_list = list(dft)
            problem_type = 'Clustering'
            classes = []
    else:
        print('Cannot find target variable to visualize. Returning...')
        return dft
    #############  Check if there are too many columns to visualize  ################
    # if len(preds) >= max_cols_analyzed:
    #     #########     In that case, SELECT IMPORTANT FEATURES HERE   ######################
    #     if problem_type.endswith('Classification') or problem_type == 'Regression':
    #         print('Number of variables = %d exceeds limit, finding top %d variables through XGBoost' %(len(
    #                                         preds), max_cols_analyzed))
    #         important_features,num_vars, _ = find_top_features_xgb(dft,preds,continuous_vars,
    #                                                      depVar,problem_type,corr_limit,verbose)
    #         if len(important_features) >= max_cols_analyzed:
    #             print('    Since number of features selected is greater than max columns analyzed, limiting to %d variables' %max_cols_analyzed)
    #             important_features = important_features[:max_cols_analyzed]
    #         dft = dft[important_features+[depVar]]
    #         #### Time to  classify the important columns again. Set verbose to zero so you don't print it again ###
    #         var_df = classify_columns(dft[important_features], verbose=0)
    #         IDcols = var_df['id_vars']
    #         discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars']
    #         cols_delete = var_df['cols_delete']
    #         bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    #         int_vars = var_df['int_vars']
    #         categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] + int_vars + bool_vars
    #         if len(var_df['continuous_vars'])==0 and len(int_vars)>0:
    #             continuous_vars = var_df['int_vars']
    #             categorical_vars = list_difference(categorical_vars, int_vars)
    #             int_vars = []
    #         else:
    #             continuous_vars = var_df['continuous_vars']
    #         date_vars = var_df['date_vars']
    #         preds = [x for x in important_features if x not in IDcols+cols_delete+discrete_string_vars]
    #         if len(IDcols+cols_delete+discrete_string_vars) == 0:
    #             print('    No variables removed since no ID or low-information variables found in data')
    #         else:
    #             print('    %d variable(s) removed since they were ID or low-information variables'
    #                                     %len(IDcols+cols_delete+discrete_string_vars))
    #         if verbose >= 1:
    #             print('    List of variables removed: %s' %(IDcols+cols_delete+discrete_string_vars))
    #         dft = dft[preds+[depVar]]
    #     else:
    #         continuous_vars = continuous_vars[:max_cols_analyzed]
    #         print('%d numeric variables in data exceeds limit, taking top %d variables' %(len(
    #                                         continuous_vars), max_cols_analyzed))
    #         if verbose >= 1:
    #             print('    List of variables selected: %s' %(continuous_vars[:max_cols_analyzed]))
    # #elif len(continuous_vars) < 1:
    # #    print('No continuous variables in this data set. No visualization can be performed')
    # #    ### Return data frame as is #####
    # #    return dfte
    # else:
        #########     If above 1 but below limit, leave features as it is   ######################
    if not isinstance(depVar, list):
        if depVar != '':
            dft = dft[preds+[depVar]]
    else:
        dft = dft[preds+depVar]
    ###################   Time to reduce cat vars which have more than 30 categories #############
    #discrete_string_vars += np.array(categorical_vars)[dft[categorical_vars].nunique()>30].tolist()
    #categorical_vars = left_subtract(categorical_vars,np.array(
    #    categorical_vars)[dft[categorical_vars].nunique()>30].tolist())
    #############   Next you can print them if verbose is set to print #########
    # ppt = pprint.PrettyPrinter(indent=4)
    # if verbose>=2 and len(cols_list) <= max_cols_analyzed:
    #     #marthas_columns(dft,verbose)
    #     print("   Columns to delete:")
    #     ppt.pprint('   %s' %cols_delete)
    #     print("   Boolean variables %s ")
    #     ppt.pprint('   %s' %bool_vars)
    #     print("   Categorical variables %s ")
    #     ppt.pprint('   %s' %categorical_vars)
    #     print("   Continuous variables %s " )
    #     ppt.pprint('   %s' %continuous_vars)
    #     print("   Discrete string variables %s " )
    #     ppt.pprint('   %s' %discrete_string_vars)
    #     print("   Date and time variables %s " )
    #     ppt.pprint('   %s' %date_vars)
    #     print("   ID variables %s ")
    #     ppt.pprint('   %s' %IDcols)
    #     print("   Target variable %s ")
    #     ppt.pprint('   %s' %depVar)
    # elif verbose==1 and len(cols_list) > 30:
    #     print('   Total columns > 30, too numerous to print.')
    return dft,depVar,IDcols,bool_vars,categorical_vars,continuous_vars,discrete_string_vars,date_vars,classes,problem_type, cols_list
####################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of printing data types and information compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols' % data.shape)
    if data.shape[1] > 30:
        print('Too many columns to print')
    else:
        if verbose>=3:
            print('Data Set columns info:')
            for col in data.columns:
                print('* %s: %d nulls, %d unique vals, most common: %s' % (
                        col,
                        data[col].isnull().sum(),
                        data[col].nunique(),
                        data[col].value_counts().head(2).to_dict()
                    ))
            print('--------------------------------------------------------------------')

#################################################################################
import copy
def EDA_find_remove_columns_with_infinity(df, remove=False):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    If remove flag is set, then it returns a smaller dataframe with inf columns removed.
    """
    nums = df.select_dtypes(include='number').columns.tolist()
    dfx = df[nums]
    sum_rows = np.isinf(dfx).values.sum()
    add_cols =  list(dfx.columns.to_series()[np.isinf(dfx).any()])
    if sum_rows > 0:
        print('    there are %d rows and %d columns with infinity in them...' %(sum_rows,len(add_cols)))
        if remove:
            ### here you need to use df since the whole dataset is involved ###
            nocols = [x for x in df.columns if x not in add_cols]
            print("    Shape of dataset before %s and after %s removing columns with infinity" %(df.shape,(df[nocols].shape,)))
            return df[nocols]
        else:
            ## this will be a list of columns with infinity ####
            return add_cols
    else:
        ## this will be an empty list if there are no columns with infinity
        return add_cols
#######################################################################################
from collections import Counter
import time
# from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
# from sklearn.feature_selection import SelectKBest
################################################################################
from collections import defaultdict
from collections import OrderedDict
import time
def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict

def count_freq_in_list(lst):
    """
    This counts the frequency of items in a list but MAINTAINS the order of appearance of items.
    This order is very important when you are doing certain functions. Hence this function!
    """
    temp=np.unique(lst)
    result = []
    for i in temp:
        result.append((i,lst.count(i)))
    return result

def find_corr_vars(correlation_dataframe,corr_limit = 0.70):
    """
    This returns a dictionary of counts of each variable and how many vars it is correlated to in the dataframe
    """
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_items = lambda dic: [x for x in dic.items()]
    a = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(a,k=1)[0],np.triu_indices_from(a,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    correlated_pair_dict = dict(correlated_pair)
    flat_corr_pair_list = [item for sublist in correlated_pair for item in sublist]
    #### You can make it a dictionary or a tuple of lists. We have chosen the latter here to keep order intact.
    #corr_pair_count_dict = Counter(flat_corr_pair_list)
    corr_pair_count_dict = count_freq_in_list(flat_corr_pair_list)
    corr_list = list(set(flatten(flatten_items(correlated_pair_dict))))
    rem_col_list = left_subtract(list(correlation_dataframe),list(OrderedDict.fromkeys(flat_corr_pair_list)))
    return corr_pair_count_dict, rem_col_list, corr_list, correlated_pair_dict
#################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
