# Cyclizability-Prediction-Website

Developers: Park Jonghan and Sophia Yan

The Cyclizability Prediction Website predicts the preference of DNA bending in 3D space.

Input format:

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/images/input.png?raw=true)

The website has two main functions: Spatial analysis and C(0,26,29,31)free predictions

## Spatial analysis

Input should be a valid PDB ID or a custom sequence + pdb file

### Example (7OHC):

#### Resulting 3D visualization

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/images/7ohc.png?raw=true)

The vector predictions can be downloaded as a .pdb file

---

#### Amplitude Graph

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/images/amplitude.png?raw=true)

The amplitude graph can be downloaded as a .png, .svg or .jpeg

The data can be downloaded as a .txt in the form of (x,y) coordinates

---

#### Phase Graph

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/images/phase.png?raw=true)

The phase graph can be downloaded as a .png, .svg or .jpeg

The data can be downloaded as a .txt in the form of (x,y) coordinates

---

## C(0,26,29,31)free predictions

Input should be a valid PDB ID or a custom sequence (length >= 50bp)

### Example (7OHC):

#### Resulting C0free prediction

![alt text](https://github.com/codergirl1106/Cyclizability-Prediction-Website/blob/main/images/c0free_prediction.png?raw=true)

The C0free prediction graph can be downloaded as a .png, .svg or .jpeg

The data can be downloaded as a .txt in the form of (50bp seqeunce segment, y) pairs

The C(26,29,31)free predictions also share the above C0free prediction functionality (not displayed here)
