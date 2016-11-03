# keras-mdn

Fitting data using neural network not with mean square error, but with a probablistic gaussian mixture model. This is an implementation of Christopher M Bishop's 1994 paper from http://eprints.aston.ac.uk/373/1/NCRG_94_004.pdf

mdn.py contains the implementation of the custom keras layer and objective function
main.py contains two test examples

Slightly different from the formula given in the paper, I have added a small epsilon term to the equation in (22) and (23) to avoid division by zero and taking log of zero

First test example is a 1d to 1d mapping
![Alt text](/screenshots/mdn_in_2d.png?raw=true "mapping 1d to 1d")
Second test example is a 1d to 2d mapping
![Alt text](/screenshots/mdn_3d.png?raw=true "mapping 1d to 2d")
