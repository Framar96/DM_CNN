# DM-CNN
Dark Matter Convolutional Neural Network (DM-CNN)

This repository documents the code developed for the Master thesis project at the GRAPPA (Gravitation and Astroparticle Physics Amsterdam) for the development and training of DM-CNN. The purpose of this neural network is to perform parameter estimation on suynthetic GW data.

The code can be split into 3 scripts to be executed sequentially.

# Installation instructions
pydd should be installed from https://github.com/adam-coogan/pydd in order to simulate dark matter spikes and generate Gravitational Wave (GW) signals.

# FILE 1 data_generator.py

This script allows to generate a synthetic dataset of GW signal frequency series in 1D arrays based on user-defined parameters using the pydd Python package.

# FILE 2 grid_search.py

This file allows to perform a grid search k-fold cross-validation to find the optimal hyperparameter values to train DM-CNN efficiently on the data generated.

# FILE 3 neural_network.py

This script trains DM-CNN on the data generated from FILE 1 and generates plots describing the model performance.

# results
The images in the results directory show the output of model evaluation on test data.
