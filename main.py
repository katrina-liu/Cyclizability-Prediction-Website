import streamlit as st

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import time 
import random 
import math
import io

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import scipy.stats
from scipy.stats import f_oneway, levene, mannwhitneyu, normaltest, ttest_ind
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split

import re
import py3Dmol
from stmol import showmol
import requests
import streamlit_scrollable_textbox as stx
import streamlit_js_eval

swidth = int(streamlit_js_eval.streamlit_js_eval(js_expressions='screen.width', want_output = True, key = 'SCR'))

root = './adapter-free-Model'

@st.cache_resource(max_entries=5)
def load_model0():
    return keras.models.load_model(root + '/C0free')

@st.cache_resource(max_entries=5)
def load_model26():
    return keras.models.load_model(root + '/C26free')

@st.cache_resource(max_entries=5)
def load_model29():
    return keras.models.load_model(root + '/C29free')

@st.cache_resource(max_entries=5)
def load_model31():
    return keras.models.load_model(root + '/C31free')

@st.cache_data(max_entries=5)
def getTexttt(pbdid): 
    link = f"https://files.rcsb.org/download/{pbdid}.pdb"
    texttt = requests.get(link).text
    return texttt

@st.cache_data(max_entries=5)
def getSequence(pbdid):
    sequencelink = f"https://www.rcsb.org/fasta/entry/{pbdid}"
    tt = requests.get(sequencelink).text
    seq = re.findall("\n[A|C|G|T]+\n", tt)[0][1:-1]
    chain = ''.join(re.findall(f">{pbdid.upper()}_\d|Chain ([A-Z])|DNA \(\d+-MER\)|synthetic construct \(\d+\)\n[A|C|G|T]+\n",tt))[0]
    otherlink = f'https://files.rcsb.org/download/{pbdid}.cif'
    texttt = requests.get(otherlink).text
    stuff = re.findall(f'ATOM[^\S\r\n]+(\d+)[^\S\r\n]+([A-Z])[^\S\r\n]+(\"?[A-Z]+\d*\'?\"?)[^\S\r\n]+\.[^\S\r\n]+([A-Z]+)[^\S\r\n]+{chain}[^\S\r\n]+([0-9]+)[^\S\r\n]+([0-9]+).+\n',texttt)
    return seq[int(stuff[0][5])-1:int(stuff[-1][5])]

model0 = load_model0()
model26 = load_model26()
model29 = load_model29()
model31 = load_model31()

def pred(model, pool): 
    input = np.zeros((len(pool), 200))
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j].upper()]] = 1
    A = model.predict(input, batch_size=128).reshape(len(pool), )
    return A

@st.cache_data(max_entries=5)
def envelope(x, y):
    x, y = list(x), list(y)
    uidx, ux, uy = [0], [x[0]], [y[0]]
    lidx, lx, ly = [0], [x[0]], [y[0]]

    # local extremas
    for i in range(1, len(x)-1):
        if (y[i] == max(y[max(0, i-3):min(i+4, len(y))])):
            uidx.append(i)
            ux.append(x[i])
            uy.append(y[i])
        if (y[i] == min(y[max(0, i-3):min(i+4, len(y))])):
            lidx.append(i)
            lx.append(x[i])
            ly.append(y[i])

    uidx.append(len(x)-1)
    ux.append(x[-1])
    uy.append(y[-1])
    lidx.append(len(x)-1)
    lx.append(x[-1])
    ly.append(y[-1])

    ubf = interp1d(ux, uy, kind=3, bounds_error=False)
    lbf = interp1d(lx, ly, kind=3, bounds_error=False)
    ub = np.array([y, ubf(x)]).max(axis=0)
    lb = np.array([y, lbf(x)]).min(axis=0)

    return ub, lb, ub-lb

# spatial analysis - amplitude and phase of DNA bending

def func(x): # x = [C0, amp, psi, c26_, c29_, c31_]
    return [c26_ - x[0] - x[1]**2*math.cos((34.5/10.3-3)*2*math.pi-math.pi*2/3 - x[2]),
            c29_ - x[0] - x[1]**2*math.cos((31.5/10.3-3)*2*math.pi-math.pi*2/3 - x[2]),
            c31_ - x[0] - x[1]**2*math.cos((29.5/10.3-2)*2*math.pi-math.pi*2/3 - x[2])]

#screenn = streamlit_js_eval.streamlit_js_eval(js_expressions='screen.width', want_output = True, key = "SCR")

#def disp_width():
#    return screenn

@st.cache_data(max_entries=5)
def show_st_3dmol(pdb_code,original_pdb,style_lst=None,label_lst=None,reslabel_lst=None,zoom_dict=None,surface_lst=None,cartoon_style="oval",
                  cartoon_radius=0.2,cartoon_color="lightgray",zoom=1,spin_on=False):
                      
    if swidth >= 1000:
        view = py3Dmol.view(width=int(swidth/2), height=int(swidth/3))
    else:
        view = py3Dmol.view(width=int(swidth), height=int(swidth))
        
    view.addModelsAsFrames(pdb_code)
    view.addModelsAsFrames(original_pdb)

    view.setStyle({"cartoon": {"style": cartoon_style,"color": cartoon_color,"thickness": cartoon_radius}})

    style_lst = []
    surface_lst = []
    
    view.addStyle({'chain':'Z'}, {'stick': {"color": "blue"}})

    view.addStyle({'chain':'A'}, {'line': {}})
    view.addStyle({'chain':'B'}, {'line': {}})
    view.addStyle({'chain':'C'}, {'line': {}})
    view.addStyle({'chain':'D'}, {'line': {}})
    view.addStyle({'chain':'E'}, {'line': {}})
    view.addStyle({'chain':'F'}, {'line': {}})
    view.addStyle({'chain':'G'}, {'line': {}})
    view.addStyle({'chain':'H'}, {'line': {}})
    view.addStyle({'chain':'I'}, {'line': {}})
    view.addStyle({'chain':'J'}, {'line': {}})
    
    view.zoomTo()
    view.spin(spin_on)
    view.zoom(zoom)

    if swidth >= 1000:
        showmol(view, height=int(swidth/3), width=int(swidth/2))
    else:
        showmol(view, height=int(swidth), width=int(swidth))

c26_, c29_, c31_ = None, None, None

@st.cache_data(hash_funcs={"MyUnhashableClass": lambda _: None})
def longcode(sequence, name, factor=30):
    global c26_, c29_, c31_
    
    pdb_output = "HEADER    output from spatial analysis\n"
    L = 200
    pool = []
    base = ['A','T','G','C']
    for i in range(L):
        left = ''.join([random.choice(base) for i in range(49)])
        right = ''.join([random.choice(base) for i in range(49)])
        pool.append(left + sequence + right)

    seqlist = dict()
    indext = 0
    for i in range(len(pool)):
        for j in range(len(pool[i])-50+1):
            tt = pool[i][j:j+50]
            if tt not in seqlist:
                seqlist.update({tt: indext})
                indext += 1

    seqlistkeys = list(seqlist.keys())
    qwer26 = pred(model26, seqlistkeys)
    qwer29 = pred(model29, seqlistkeys)
    qwer31 = pred(model31, seqlistkeys)

    c26 = np.zeros((L, len(pool[0])-50+1))
    c29 = np.zeros((L, len(pool[0])-50+1))
    c31 = np.zeros((L, len(pool[0])-50+1))

    for i in range(len(pool)):
        for j in range(49):
            c26[i,j] = qwer26[seqlist[pool[i][j:j+50]]]
            c29[i,j] = qwer29[seqlist[pool[i][j:j+50]]]
            c31[i,j] = qwer31[seqlist[pool[i][j:j+50]]]
        for j in range(len(pool[i])-50+1-49, len(pool[i])-50+1):
            c26[i,j] = qwer26[seqlist[pool[i][j:j+50]]]
            c29[i,j] = qwer29[seqlist[pool[i][j:j+50]]]
            c31[i,j] = qwer31[seqlist[pool[i][j:j+50]]]

    c26 = c26.mean(axis=0)
    c29 = c29.mean(axis=0)
    c31 = c31.mean(axis=0)

    seqlist = [sequence[i:i+50] for i in range(len(sequence)-50+1)]
    if(len(sequence) >= 50):
        c26[49:-49] = pred(model26, seqlist).reshape(-1, )
        c29[49:-49] = pred(model29, seqlist).reshape(-1, )
        c31[49:-49] = pred(model31, seqlist).reshape(-1, )

    c26 -= c26.mean()
    c29 -= c29.mean()
    c31 -= c31.mean()

    zxcv26 = c26
    zxcv29 = c29
    zxcv31 = c31
    u26, l26, caf26 = envelope(np.arange(len(zxcv26)), zxcv26)
    u29, l29, caf29 = envelope(np.arange(len(zxcv29)), zxcv29)
    u31, l31, caf31 = envelope(np.arange(len(zxcv31)), zxcv31)
    amp = (caf26 + caf29 + caf31)/3

    psi = []
    for i in range(len(amp)):
        c26_, c29_, c31_ = zxcv26[i], zxcv29[i], zxcv31[i]
        root = fsolve(func, [1, 1, 1])
        psi.append(root[2])
        if(psi[-1] > math.pi): psi[-1] -= 2*math.pi
    psi = np.array(psi)

    # trim random sequences
    amp = amp[25:-25]
    psi = psi[25:-25]

    lines = re.findall('ATOM[^\S\r\n]+(\d+)[^\S\r\n]+(C8|C6)[^\S\r\n]+(DA|DG|DC|DT)[^\S\r\n]+([A-Z])[^\S\r\n]+(-?[0-9]+)[^\S\r\n]+'+"(-?\d+[\.\d]*)[^\S\r\n]*"*5+"([A-Z])",name)

    # find helical axis
    qwer = []
    for line in lines:
        if line[2] in ['DA', 'DG'] and line[1] == 'C8':
            qwer.append([float(line[_]) for _ in range(5, 8)])
        if line[2] in ['DT', 'DC'] and line[1] == 'C6':
            qwer.append([float(line[_]) for _ in range(5, 8)])
            
    qwer, asdf = np.array(qwer[:len(qwer)//2]), np.array(qwer[-(len(qwer)//2):][::-1])
    
    otemp = (qwer+asdf)/2
    o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)])
    ytemp = qwer - asdf
    z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)])
    ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
    x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))])
    y = np.array([np.cross(z[i], x[i]) for i in range(len(z))])

    x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))]) # direction of minor groove
    y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))])
    z = np.array([z[i]/np.linalg.norm(z[i]) for i in range(len(z))])

    arrow = []
    for j in range(len(x)): arrow.append(np.cos(-psi[j])*x[j] + np.sin(-psi[j])*y[j])
    arrow = np.array(arrow)
    for j in range(len(arrow)):
        arrow[j] /= np.linalg.norm(arrow[j])
        arrow[j] *= factor*amp[j]
    e = o + arrow
    o = np.around(o, 3)
    e = np.around(e, 3)

    for j in range(len(o)):
        pdb_output += 'HETATM' + str(j+1).rjust(5) + ' C    AXI ' + 'Z' + '   1    ' # '    1' 5d
        for k in range(3):
            pdb_output += str(o[j][k]).rjust(8)
        pdb_output += '\n'
    for j in range(len(e)):
        pdb_output += 'HETATM' + str(j+len(o)+1).rjust(5) + ' C    AXI ' + 'Z' + '   1    ' # '    1' 5d
        for k in range(3):
            pdb_output += str(round(e[j][k], 2)).rjust(8)
        pdb_output += '\n'
    for j in range(len(o)-1):
        pdb_output += 'CONECT' + str(j+1).rjust(5) + str(j+2).rjust(5) + '\n' # CONECT    1    2
    for j in range(len(o)):
        pdb_output += 'CONECT' + str(j+1).rjust(5) + str(j+len(o)+1).rjust(5) + '\n' # CONECT    1    2

    return pdb_output, amp, psi

st.title("Cyclizability Prediction\n")
st.markdown("---")
st.subheader("website guide")
st.markdown("***functions*** + ***parameters***")
st.markdown("***:blue[spatial visualization function]***: ***:green[pdb id]*** OR ***:violet[custom sequence + pdb file]***")
st.markdown("***:blue[C0, C26, C29, or C31 predictions]***: ***:green[pdb id (fasta sequence >= 50bp)]*** OR ***:violet[custom sequence (>= 50bp)]***")
st.markdown("---")

col1, col2, col3 = st.columns([0.46, 0.08, 0.46])
seq = ''
with col1:
    seq = st.text_input('input a sequence', seq)
    texttt = ''
with col2:
    st.subheader("OR")
with col3:
    pdbid = st.text_input('PDB ID','7OHC').upper()
    texttt=''
    if pdbid != '' and seq == '':
        try:
            texttt = getTexttt(pdbid)
            seq = getSequence(pdbid)
        except:
            pass
    uploaded_file = st.file_uploader("upload a pdb file")

    if uploaded_file is not None:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))        
        texttt = stringio.read()
        
st.markdown("---")
st.subheader("Please select the parameter you would like to predict/view")
option = st.selectbox('', ('Spatial analysis', 'C0free prediction', 'C26free prediction', 'C29free prediction', 'C31free prediction'))

if option == 'Spatial analysis' and seq != '' and texttt != '':
    st.markdown("***")
    st.header(f"Spatial Visualization")
    
    factor = st.text_input('vector length scale factor','e.g. 30')
    try:
        pdb_output, amp, psi = longcode(seq, texttt, int(factor))
    except:
        pdb_output, amp, psi = longcode(seq, texttt)

    figg, axx = plt.subplots()
    figgg, axxx = plt.subplots()
    
    plt.figure(figsize=(10, 3))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    smth1 = np.arange(0.5, 0.5+len(amp))
    smth2 = np.arange(0.5, 0.5+len(psi))
    axx.plot(smth1, amp)
    axxx.plot(smth2, psi)
    
    file_nameu = st.text_input('file name', 'e.g. spatial_visualization.pdb')
    show_st_3dmol(pdb_output,texttt)
    st.download_button('Download .pdb', pdb_output, file_name=f"{file_nameu}")
            
    st.markdown("***")
    st.header(f"Amplitude Graph")
    filetype = st.selectbox('amplitude graph file type', ('svg', 'png', 'jpeg'))
            
    imgg = io.BytesIO()
    figg.savefig(imgg, format=filetype)
    file_name3 = st.text_input('file name', f'e.g. amplitude_graph.{filetype}')
    btn3 = st.download_button(label="Download graph",data=imgg,file_name=f"{file_name3}",mime=f"image/{filetype}")
    st.pyplot(figg)

    st.markdown("***")
    
    st.header(f"Data for Amplitude Graph")
    long_text11 = ""
    for i in range(len(amp)):
        long_text11 += f"{smth1[i]}, {amp[i]}\n"

    file_name11 = st.text_input('file name', f'e.g. amplitude_data.txt')
    
    st.download_button('Download data', long_text11, file_name=f"{file_name11}")

    st.markdown("data format in (x, y) coordinates")
    stx.scrollableTextbox(long_text11, height = 300)
    
    st.markdown("***")
            
    st.header(f"Phase Graph")
    filetype2 = st.selectbox('phase graph file type', ('svg', 'png', 'jpeg'))
            
    imggg = io.BytesIO()
    figgg.savefig(imgg, format=filetype2)
    file_name4 = st.text_input('file name', f'e.g. phase_graph.{filetype2}')
    btn4 = st.download_button(label="Download graph",data=imggg,file_name=f"{file_name4}",mime=f"image/{filetype2}")
    st.pyplot(figgg)

    st.markdown("***")
    
    st.header(f"Data for Phase Graph")
    
    long_text22 = ""
    for i in range(len(psi)):
        long_text22 += f"{smth2[i]}, {psi[i]}\n"

    file_name22 = st.text_input('file name', f'e.g. phase_data.txt')
    
    st.download_button('Download data', long_text22, file_name=f"{file_name22}")

    st.markdown("data format in (x, y) coordinates")
    stx.scrollableTextbox(long_text22, height = 300)
elif option == 'Spatial analysis' and texttt == '' and seq != '':
    st.subheader(":red[Please attach a pdb file to visualize]")
elif len(seq) >= 50:
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
    st.header(f"Graph of C{model[5:]}free prediction")
    filetype5 = st.selectbox('file type', ('png', 'svg', 'jpeg'))
    img = io.BytesIO()
    fig.savefig(img, format=filetype5)

    file_name1 = st.text_input('file name', f'e.g. {option.replace(" ", "_")}.{filetype5}')
    btn = st.download_button(
            label="Download graph",
            data=img,
            file_name=f"{file_name1}",
            mime=f"image/{filetype5}"
    )
    
    st.pyplot(fig)
    st.markdown("***")
    st.header(f"Data of C{model[5:]}free prediction")
    # show data in scrollable window
    long_text = ""
    for i in range(len(cNfree)):
        long_text += f"{list50[i]} {cNfree[i]}\n"

    file_name = st.text_input('file name', f'e.g. {option.replace(" ", "_")}.txt')
    
    st.download_button('Download data', long_text, file_name=f"{file_name}")
    
    stx.scrollableTextbox(long_text, height = 300)
else:
    st.subheader(":red[Please provide a sequence (>= 50bp) or a pdb id/file]")
