import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import random 
import math
import io
import time
from tensorflow import keras

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import re
import py3Dmol
from stmol import showmol
import requests
import streamlit_scrollable_textbox as stx
import streamlit_js_eval
import functools

swidth = streamlit_js_eval.streamlit_js_eval(js_expressions='screen.width', want_output = True, key = 'SCR')
try:
    swidth = int(swidth)
except TypeError:
    swidth = 1100

def my_cache(f):
    @st.cache_data(max_entries=5, ttl=600)
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)
    return inner

@st.cache_data
def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")

def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128)
    A.resize((len(pool),))
    return A

def getTexttt(pbdid): 
    link = f"https://files.rcsb.org/download/{pbdid}.pdb"
    texttt = requests.get(link).text
    return texttt

def getSequence(pbdid):
    sequencelink = f"https://www.rcsb.org/fasta/entry/{pbdid}"
    tt = requests.get(sequencelink).text
    seq = re.findall("\n[A|C|G|T]+\n", tt)[0][1:-1]
    chain = ''.join(re.findall(f">{pbdid.upper()}_\d|Chains? ([A-Z]).*\n[A|C|G|T]+\n",tt))[0]
    otherlink = f'https://files.rcsb.org/download/{pbdid}.cif'
    tt = requests.get(otherlink).text
    stuff = re.findall(f'ATOM[^\S\r\n]+(\d+)[^\S\r\n]+([A-Z])[^\S\r\n]+(\"?[A-Z]+\d*\'?\"?)[^\S\r\n]+\.[^\S\r\n]+([A-Z]+)[^\S\r\n]+{chain}[^\S\r\n]+([0-9]+)[^\S\r\n]+([0-9]+).+\n',tt)
    return seq[int(stuff[0][5])-1:int(stuff[-1][5])]

def envelope(fity):
    ux, uy = [0], [fity[0]]
    lx, ly = [0], [fity[0]]
    
    # local extremas
    for i in range(1, len(fity)-1):
        if (fity[i] == max(fity[max(0, i-3):min(i+4, len(fity))])):
            ux.append(i)
            uy.append(fity[i])
        if (fity[i] == min(fity[max(0, i-3):min(i+4, len(fity))])):
            lx.append(i)
            ly.append(fity[i])

    ux.append(len(fity)-1)
    uy.append(fity[-1])
    lx.append(len(fity)-1)
    ly.append(fity[-1])
    
    ub = np.array([fity, interp1d(ux, uy, kind=3, bounds_error=False)(range(len(fity)))]).max(axis=0)
    lb = np.array([fity, interp1d(lx, ly, kind=3, bounds_error=False)(range(len(fity)))]).min(axis=0)
    return ub-lb

def trig(x, *args): # x = [C0, amp, psi]
    return [args[0][0] - x[0] - x[1]**2*math.cos((34.5/10.3-3)*2*math.pi-math.pi*2/3 - x[2]),
            args[0][1] - x[0] - x[1]**2*math.cos((31.5/10.3-3)*2*math.pi-math.pi*2/3 - x[2]),
            args[0][2] - x[0] - x[1]**2*math.cos((29.5/10.3-2)*2*math.pi-math.pi*2/3 - x[2])]

def show_st_3dmol(pdb_code,original_pdb,cartoon_style="oval",
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
    return 0

def helpercode(model_num: int, seqlist: dict, pool, sequence):
    prediction = pred(load_model(model_num), tuple(seqlist.keys()))
    
    result_array = np.zeros((len(pool), len(pool[0]) - 49), dtype = np.single)

    for i in range(len(pool)):
        for j in range(49):
            result_array[i, j] = prediction[seqlist[pool[i][j:j + 50]]]
        for j in range(len(pool[i])-98, len(pool[i])-49):
            result_array[i, j] = prediction[seqlist[pool[i][j:j + 50]]]

    result_array = result_array.mean(axis=0)

    seqlist = [sequence[i:i+50] for i in range(len(sequence)-49)]
    result_array[49:-49] = pred(load_model(model_num), seqlist).reshape(-1, )

    result_array -= result_array.mean()
    return result_array

def pdb_out(name, psi, amp, factor):
    lines = re.findall('ATOM[^\S\r\n]+(\d+)[^\S\r\n]+(C8|C6)[^\S\r\n]+(DA|DG|DC|DT)[^\S\r\n]+([A-Z])[^\S\r\n]+(-?[0-9]+)[^\S\r\n]+'+"(-?\d+[\.\d]*)[^\S\r\n]*"*5+"([A-Z])",name)

    # find helical axis
    qwer = []
    for line in lines:
        if line[2] in ['DA', 'DG'] and line[1] == 'C8':
            qwer.append([float(line[_]) for _ in range(5, 8)])
        if line[2] in ['DT', 'DC'] and line[1] == 'C6':
            qwer.append([float(line[_]) for _ in range(5, 8)])
            
    qwer, asdf = np.array(qwer[:len(qwer)//2], dtype = np.single), np.array(qwer[-(len(qwer)//2):][::-1], dtype = np.single)
    
    otemp = (qwer+asdf)/2
    o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)
    ytemp = qwer - asdf
    z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
    ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
    x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
    y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)

    x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
    y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
    z = np.array([z[i]/np.linalg.norm(z[i]) for i in range(len(z))], dtype = np.single)

    arrow = []
    for j in range(len(x)): arrow.append(np.cos(-psi[j])*x[j] + np.sin(-psi[j])*y[j])
    arrow = np.array(arrow)
    for j in range(len(arrow)):
        arrow[j] /= np.linalg.norm(arrow[j])
        arrow[j] *= factor*amp[j]
    e = o + arrow
    o = np.around(o, 3)
    e = np.around(e, 3)

    pdb_output = "HEADER    output from spatial analysis\n"
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

    return pdb_output

@my_cache
def longcode(sequence, name, factor):
    pool = []
    base = ['A','T','G','C']
    for i in range(200):
        left = ''.join([random.choice(base) for i in range(49)])
        right = ''.join([random.choice(base) for i in range(49)])
        pool.append(left + sequence + right)

    seqlist = dict()
    indext = 0
    for i in range(len(pool)):
        for j in range(len(pool[i])-49):
            tt = pool[i][j:j+50]
            if tt not in seqlist:
                seqlist.update({tt: indext})
                indext += 1
    
    models = dict.fromkeys((26, 29, 31))
    for modelnum in models.keys():
        models[modelnum] = helpercode(modelnum, seqlist, pool, sequence)

    amp = sum(envelope(m) for m in models.values()) / len(models)
    
    psi = []
    for i in range(len(amp)):
        root = fsolve(trig, [1, 1, 1], args=[m[i] for m in models.values()])
        psi.append(root[2])
        if(psi[-1] > math.pi): psi[-1] -= 2*math.pi
    psi = np.array(psi, dtype = np.single)

    # trim random sequences
    amp,psi = amp[25:-25],psi[25:-25]

    return pdb_out(name, psi, amp, factor), amp, psi

def spatial_analysis_ui(imgg, seq, texttt):
    st.markdown("***")
    st.header(f"Spatial Visualization")
        
    factor = st.text_input('vector length scale factor','30')
    try:
        factor = int(factor)
    except:
        factor = 30

    pdb_output, amp, psi = longcode(seq, texttt, factor)
        
    figg, axx = plt.subplots()
    figgg, axxx = plt.subplots()
        
    plt.figure(figsize=(10, 3))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    axx.plot(amp)
    axxx.plot(psi)
    
    file_nameu = st.text_input('file name', 'spatial_visualization.pdb')
    show_st_3dmol(pdb_output,texttt)
    st.download_button('Download .pdb', pdb_output, file_name=f"{file_nameu}")
                
    st.markdown("***")
    st.header(f"Amplitude Graph")
    filetype = st.selectbox('amplitude graph file type', ('svg', 'png', 'jpeg'))
        
    figg.savefig(imgg, format=filetype)
    file_name3 = st.text_input('file name', f'amplitude_graph.{filetype}')
    btn3 = st.download_button(label="Download graph",data=imgg,file_name=f"{file_name3}",mime=f"image/{filetype}")
    st.pyplot(figg)

    st.markdown("***")
        
    st.header(f"Data for Amplitude Graph")
    long_text11 = "\n".join(f"{0.5+i}, {amp[i]}" for i in range(len(amp)))

    file_name11 = st.text_input('file name', f'amplitude_data.txt')
        
    st.download_button('Download data', long_text11, file_name=f"{file_name11}")

    st.markdown("data format in (x, y) coordinates")
    stx.scrollableTextbox(long_text11, height = 300)
        
    st.markdown("***")
                
    st.header(f"Phase Graph")
    filetype2 = st.selectbox('phase graph file type', ('svg', 'png', 'jpeg'))
        
    figgg.savefig(imgg, format=filetype2)
    file_name4 = st.text_input('file name', f'phase_graph.{filetype2}')
    btn4 = st.download_button(label="Download graph",data=imgg,file_name=f"{file_name4}",mime=f"image/{filetype2}")
    st.pyplot(figgg)

    st.markdown("***")
        
    st.header(f"Data for Phase Graph")
        
    long_text22 = "\n".join(f"{0.5+i}, {psi[i]}" for i in range(len(psi)))

    file_name22 = st.text_input('file name', f'phase_data.txt')
        
    st.download_button('Download data', long_text22, file_name=f"{file_name22}")

    st.markdown("data format in (x, y) coordinates")
    stx.scrollableTextbox(long_text22, height = 300)

def sequence_ui(imgg, seq, option):
    list50 = [seq[i:i+50] for i in range(len(seq)-50+1)]

    modelnum = int(re.findall(r'\d+', option)[0])

    cNfree = list(pred(load_model(modelnum), list50))
    
    # show matplotlib graph
    fig, ax = plt.subplots()
    ax.plot(cNfree)
    plt.figure(figsize=(10, 3))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # download matplotlib graph
    st.markdown("***")
    st.header(f"Graph of C{modelnum}free prediction")
    filetype5 = st.selectbox('file type', ('png', 'svg', 'jpeg'))
        
    fig.savefig(imgg, format=filetype5)

    file_name1 = st.text_input('file name', f'C{modelnum}free_prediction.{filetype5}')
    btn = st.download_button(
        label="Download graph",
        data=imgg,
        file_name=f"{file_name1}",
        mime=f"image/{filetype5}"
    )
        
    st.pyplot(fig)
    st.markdown("***")
    st.header(f"Data of C{modelnum}free prediction")
    # show data in scrollable window
    long_text = "\n".join(f"{list50[i]} {cNfree[i]}" for i in range(len(cNfree)))

    file_name = st.text_input('file name', f'C{modelnum}free_prediction.txt')
    
    st.download_button('Download data', long_text, file_name=f"{file_name}")
        
    stx.scrollableTextbox(long_text, height = 300)

def main():
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
        seq = st.text_input('input a sequence', seq).upper()
        
        texttt = st.file_uploader("upload a pdb file")
        if texttt is not None:
            texttt = texttt.getvalue().decode("utf-8")
            
    with col2:
        st.subheader("OR")
    with col3:
        pdbid = st.text_input('PDB ID','').upper()
        texttt=None
        if pdbid != '' and seq == '':
            try:
                texttt = getTexttt(pdbid)
                seq = getSequence(pdbid)
            except:
                pass
            
    st.markdown("---")
    st.subheader("Please select the parameter you would like to predict/view")
    option = st.selectbox('', ('Spatial analysis', 'C0free prediction', 'C26free prediction', 'C29free prediction', 'C31free prediction'))

    imgg = io.BytesIO()
    if option == 'Spatial analysis' and seq != '' and texttt != None:
        spatial_analysis_ui(imgg, seq, texttt)
    elif option == 'Spatial analysis' and texttt == None and seq != '':
        st.subheader(":red[Please attach a pdb file to visualize]")
    elif len(seq) >= 50:
        sequence_ui(imgg, seq, option)
    else:
        st.subheader(":red[Please provide a sequence (>= 50bp) or a pdb id/file]")
    return 0

main()
