import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function

#how to work with .mat files in python.
#training set of handwritten digits
data = 'ex3data1.mat'
mat = scipy.io.loadmat(data)

X, Y= mat['X'], mat['y']

#X[1][:]    1first row, which represents a training example:
#each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.

#1.2 Visualizing the data

def display_data(X,Y,sample_rows=400):
    row_len_X=len(X)

    random_index=random.sample(range(0,4000), sample_rows)
    X_row=X[1000,:]
    X_row_matrix=np.reshape(X_row,(20,20))
    img=scipy.misc.toimage(X_row_matrix.T)

    for i in range(0,rows):

        random_row=X[i][:]
        for j in range(0,rows):
