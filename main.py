import streamlit as st

# cd '/Users/jingli/Dropbox (BCH)/Sophia/Sophia tj new project'
# streamlit run main.py

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import time 
import random 
import math 
from tqdm import tqdm 

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#from keras.utils import np_utils

import scipy.stats
from scipy.stats import f_oneway, levene, mannwhitneyu, normaltest, ttest_ind
from sklearn.model_selection import train_test_split

import re
import streamlit_scrollable_textbox as stx

root = './adapter-free-Model'
model0 = keras.models.load_model(root + '/C0free')
model26 = keras.models.load_model(root + '/C26free')
model29 = keras.models.load_model(root + '/C29free')
model31 = keras.models.load_model(root + '/C31free')

def pred(model, pool): 
    input = np.zeros((len(pool), 200))
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j].upper()]] = 1
    A = model.predict(input, batch_size=128).reshape(len(pool), )
    return A

st.write("""
# Cyclizability Prediction
""")

seq = st.text_input('sequence', 'GTAGC...') # seq = 'AGTTC...' ask user for it

option = st.selectbox('', ('C0free prediction', 'C26free prediction', 'C29free prediction', 'C31free prediction'))

if len(seq) >= 50:
    list50 = [seq[i:i+50] for i in range(len(seq)-50+1)]

    model = "model"+re.findall(r'\d+', option)[0]
    cNfree = pred(eval(model), list50) # model{n} for c{n}free

    # show matplotlib graph
    fig, ax = plt.subplots()
    ax.plot(list(cNfree))
    plt.figure(figsize=(10, 3))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # download matplotlib graph
    st.markdown("***")
    if st.button('download graph'):
        fig.savefig(f'{file_name}.png')

    file_name = st.text_input('file name', 'e.g. cyc_trial_6_graph')
    
    st.pyplot(fig)
    st.markdown("***")
    # show data in scrollable window
    long_text = ""
    for i in range(len(cNfree)):
        long_text += f"{list50[i]} {cNfree[i]}\n"

    if st.button('download data'):
        with open(file_name+'.txt', 'w')as a:
            a.write(long_text)

    file_name = st.text_input('file name', 'e.g. cyc_trial_6')
    
    stx.scrollableTextbox(long_text, height = 100)
