#!/usr/bin/env python

"""
read_fasta.py: Supporting prediction of DNA sequence cyclizability 
                  from fasta file
"""

import matplotlib.pyplot as plt
import numpy as np
import random 
import math
import io
import time
from tensorflow import keras
from Bio import SeqIO
import pandas as pd

import sys
import os

def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128)
    A.resize((len(pool),))
    return A

def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")

def pred_seqs(seqs):
    list50 = []
    for seq in seqs:
        list50.extend([seq[i:i+50] for i in range(len(seq)-50+1)])
    cNfree = pred(load_model(0), list50)
    cNfree.resize((len(seqs),len(seqs[0])-50+1))
    return cNfree
    
    
def read_fasta_pred(input_file, output_file):
    cyc = []
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    seqs = []
    names = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        seqs.append(sequence)
        names.append(name)
    cyc = pred_seqs(seqs)
    
    df = pd.DataFrame(data=cyc, index=names)
    df.to_csv(output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Require two positional arguments")
    else:
        read_fasta_pred(sys.argv[1], sys.argv[2])
    
