from hypothesis import given, strategies as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
import numpy as np
import imageio

from functions import *


# %% TESTING find_drop()
@given(len_plateau = st.integers(min_value=50, max_value=950))
def test_find_drop_accuracy(len_plateau):   
    """
    this is a test that verifies if the found drop is accurate with low noise level

    GIVEN: the length  of the plateau len_plateau
    WHEN: I find drop
    THEN: the result is close to len_plateau

    """
    #create exponential like array with a plateau before
    len_drop = 1000 # number of points of the drop
    
    plateau = np.ones(len_plateau)
    t = np.arange(0,5,5/len_drop)
    exp = np.exp(-(t)**0.5)
    dat = np.append(plateau,exp)

    # add noise 
    level = 0.02
    noise = np.random.normal(0, level, len(dat))
    dat = dat + noise


    drop = find_drop(dat, 0.5, False, False)
    assert np.abs(len_plateau-drop) <= 10

@given(level = st.floats(min_value=0.02, max_value=0.1))
def test_find_drop_noise(level):   
    """
    this is a test that verifies if the found drop is accurate with varying noise level

    GIVEN: the level of noise
    WHEN: I find drop
    THEN: the result is close to len_plateau

    """
    #create exponential like array with a plateau before
    len_drop = 1000 # number of points of the drop
    len_plateau = 300
    plateau = np.ones(len_plateau)
    t = np.arange(0,5,5/len_drop)
    exp = np.exp(-(t)**0.5)
    dat = np.append(plateau,exp)

    # add noise 
    noise = np.random.normal(0, level, len(dat))
    dat = dat + noise


    drop = find_drop(dat, 0.7, False, False)
    assert np.abs(len_plateau-drop) <= 20


# def test_find_drop_local_noise(position):   
#     """
#     this is a test that verifies if the found drop is accurate with a local big noise

#     GIVEN: the position of the big noise
#     WHEN: I find drop
#     THEN: the result is close to len_plateau

#     """
#     #create exponential like array with a plateau before
#     len_drop = 1000 # number of points of the drop
#     len_plateau = 300

#     plateau = np.ones(len_plateau)
#     t = np.arange(0,5,5/len_drop)
#     exp = np.exp(-(t)**0.5)
#     dat = np.append(plateau,exp)

#     # add noise 
#     level = 0.03
#     local_noise = np.random.normal(0, 0.08, 50)
    
#     #position local noise in a random position in between 50 and 950
#     noise = np.random.normal(0,level,len(dat))
#     for i in range(len(dat)):
#         if i == position : 
#             for j in range(len(local_noise)):
#                 noise[i+j] += local_noise[j]
#     dat = dat + noise


#     drop = find_drop(dat, 0.4, False, False)
#     assert np.abs(len_plateau-drop) <= 20




 TESTING intensity_in_image()

@given(int = st.integers(min_value=0, max_value=200))
def test_intensityi_in_image(int):
    """
    creates a tif image in a known path
    The image is 30x30 squares. the 30x10 middle line have values of 100 the others 1
    Averaging the ones in the middle the value should be 100
    """
    # Create an array representing the image
    image_array = np.zeros((30, 30), dtype=np.uint8)  # Initialize a 3x3 array with zeros
    image_array[10:20] = int  # Set the middle row to 100

    # Save the array as a TIFF image
    imageio.imwrite('image.tif', image_array)

    assert intensity_in_image('image.tif',[0.35, 0.66], True) == int



# # %% TESTING intensity_in_multipage_image()
# """
# Creates a multipage tif with 30x30 pixels . The values of the pixels are 0,1,2. 
# I should be [0,1,2] 
# """
# num = np.random.randint(0, 100, 3)
# # Create image arrays for each page
# page1 = np.zeros((30, 30), dtype=np.uint8) * num[0]
# page2 = np.ones((30, 30), dtype=np.uint8) *  num[1]
# page3 = np.ones((30, 30), dtype=np.uint8) * num[2]
# # Create a list of images to be written to the multi-page TIFF
# image_list = [page1, page2, page3]

# # Save the list of images as a multi-page TIFF
# imageio.mimwrite('multipage_image.tif', image_list, format='tif')

# assert  intensity_in_multipage_image('multipage_image.tif',[0.35, 0.66]).all() == num.all()




# # %% perform_st_exp_fit(t,OR,begin_drop,fit_len, st_exponential, plot_fit, fps, title)

# """
# create some data simulating a noisy exponential and fit 
# """
# def st_exp(x,b,a):
#     return np.exp(-(b*x)**a)

# #create exponential like array with a plateau before
# len_drop = 1000 # number of points of the drop
# len_plateau = 300 # number of points of the plateu
# fps = 300
# plateau = np.ones(len_plateau)
# t = np.arange(0,5,1/fps)
# exp = np.exp(-(6*t)**0.5)
# dat = np.append(plateau,exp)

# # add noise 
# level = 0.05
# noise = np.random.normal(0, level, len(dat))
# dat = dat + noise

# ret =  perform_st_exp_fit(t, dat, 300, len_drop, st_exp, True, fps, 'title')
# d = ret[0]
# alpha = ret[1]

# print('is ',d ,' = 1')
# print('is ', alpha,' = 0.5')



# # %%
