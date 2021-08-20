# Neureye
Neural network based eye calibration using spikes as used in Yates et al., 2020

## Overview
This repo contains the model code and examples to fit calibration matrices / shifters for correcting the eye-calibration using spikes from V1. An explanation of the model architecture can be found here with more background [here](https://jake.vision/blog/lurz-paper). 
![Model](./docs/model.png "Model Schematic")

The model is fit end-to-end from stimulus to spikes by minimizing the poisson loss. The parameters of the neural network and readout are really not the focus of this model. The goal is to get calibration matrices, such as these:
![CalibMats](./docs/calib.png "Calibration Matrices")

These say how much to shift the eye position as a function of where the Marmoset is looking on the screen. The x and y axis are in degrees of visual angle and the colorbar is in arcminutes.


The result of this kind of correction makes a big difference in the measured receptive fields for the population of neurons recorded. Below, you'll find the spatial receptive fields for three example neurons.
![STAs](./docs/examples.png "Calibration Matrices")

## Getting started
1. Setting up the environment...
2. Getting the example dataset...
3. Example notebook...


