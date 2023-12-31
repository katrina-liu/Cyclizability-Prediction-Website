# Cyclizability-Prediction-Website

**Maintaintained by** Sophia Yan & Jonghan Park 

**Cite**: 

J Park, B Cieza, S Yan, A Basu, & T Ha. 2024. Probing mechanical selection from population genetics data through accurate prediction of DNA mechanics. (To be published)

This builds on our previous work on measuring DNA mechanics on the genome scale (Basu, Nature. 2021) and deciphering the mechanical code (Basu, NSMB. 2022). 

## Introduction: On predictions

**Cyclizability** is a propensity of cyclization of a short DNA sequence (mostly 50 bp) measured by loop-seq (Basu, Nature. 2021), which is a proxy for bendability. 

**The model's accuracy**, evaluated by Pearson's R between measured and predicted cyclizability, is over **0.95**, which is an improvement from accuracies of previous models (~0.75). The accuracy of prediction is comparable to the precision of loop-seq experiment.

Predictions are based on 50 bp DNA sequences. For DNA > 50 bp, we slide 50 bp windows with a stride of 1 bp. 

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/accuracy.png?raw=true)

## Why adapter-free cyclizability? 

Loop-seq requires an additional pair of adapters enclosing the target 50 bp sequence, which alters cyclizability values and obscures the true effect by the target DNA alone. 

Adapter-free cyclizability is a quantity deduced by removing the effects of adapters, which is denoted as C0free (or as C26free, C29free, or C31free, when measuring DNA bending towards a specific direction). 

A pair of **reverse complementary DNA sequence have similar C0free values** as a consequence (Pearson's R ~ 0.966).

## Overview: asymmetric mechanics of Widom 601 DNA 

As understood by single-molecule experiments (Ngo, Cell. 2015; iBiology lecture series by Taekjip Ha), the left hand side (LH) of Widom 601 DNA has a stronger affinity of wrapping around the histone core particle compared to the right hand side (RH). 

**C0free** summarizes the overall bending of the Widom 601 DNA. 

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/601 c0free.png?raw=true) 

**Spatial analysis** summarizes the most probable direction of DNA bending. 

You will see the preferred direction of DNA bending in black lines, deduced from the DNA sequence alone, but matches the DNA bending in the overlaid PDB structure.

LH intrinsically bends towards the nucleosome center, but RH does not, which allowed easier unwrapping of RH by optical tweezers (Ngo, Cell. 2015). 

High C0free and consistent direction of DNA bending in 601 LH DNA are two main factors that may have contributed to the widespread of Widom 601 DNA in structural biology. In reverse, one can design an arbitrary DNA sequence that fits to a desired crystal structure. 

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/601 spatial.png?raw=true) 

## Website

The website can visualize the results of spatial analysis or C(0,26,29,31)free predictions. 

### 1.1. Spatial analysis

Input should be a valid PDB ID or a custom sequence + pdb file

#### Example (7OHC - Widom 601 DNA wrapping around a histone complex):

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/7ohc.png?raw=true)

The vector predictions can be downloaded as a .pdb file

---

### 1.2. Amplitude 

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/amplitude.png?raw=true)

Amplitude is a length of each line in the visualized spatial analysis. 

The amplitude graph can be downloaded as a .png, .svg or .jpeg

The data can be downloaded as a .txt in the form of (x,y) coordinates

---

### 1.3. Phase 

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/phase.png?raw=true)

Phase is a direction of each line in the visualized spatial analysis. 

The phase graph can be downloaded as a .png, .svg or .jpeg

The data can be downloaded as a .txt in the form of (x,y) coordinates

---

### 2. C(0,26,29,31)free predictions

Input should be a valid PDB ID or a custom sequence (length >= 50bp)

#### Example (7OHC - Widom 601 DNA wrapping around a histone complex):

#### Resulting C0free prediction

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/c0free_prediction.png?raw=true)

The C0free prediction graph can be downloaded as a .png, .svg or .jpeg

The data can be downloaded as a .txt in the form of (50bp seqeunce segment, y) pairs

The C(26,29,31)free predictions also share the above C0free prediction functionality (not displayed here)
