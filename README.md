# cv2018
Artistic Style Transfer

Jessie Liu, Sagar Lal, Richa Gode

Columbia University, COMSW4731, Computer Vision

Fall 2018

This codebase seeks to research and test out the functionality of texture synthesis, a method of transferring art style to an unvaried content image. We chose to adopt this approach because it does not utilize a convolutional neural net; rather, the algorithm is iterative and builds the combination of style and content image step by step. To improve the algorithm, we look into segmentation and color-transfer, two of the central aspects of the algorithm established by Elad and Milanafar et. al. For segmentation, we study the differences between a Canny filter and a Laplacian of Gaussians filter, in order to preserve sections of the content image that do not receive the style transfer. For color transferring, we experiment with histogram matching between the style and and content, as well as by manipulating the images in $l\alpha\beta$ space.

While still a work in progress, a first iteration of texture synthesis (ie without a true segmentation mask) can be processed by running master.py.

Code to create and save different segmentation masks is in the master.ipynb file.

We'd like to acknowledge the work of Elad and Milanafar et. al., as well as the following codebase in Matlab that served as a set of guidelines for our work: https://github.com/ewang314/EE368_Final_Project
