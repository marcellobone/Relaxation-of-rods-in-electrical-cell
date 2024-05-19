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

THRESHOLD = 1e-1


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



@given(position = st.integers(min_value=0, max_value=1250))
def test_find_drop_local_noise(position):   
    """
    this is a test that verifies if the found drop is accurate with a local big noise

    GIVEN: the position of the big noise
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
    level = 0.03
    local_noise = np.random.normal(0, 0.08, 50)
    
    #position local noise in a random position in between 50 and 950
    noise = np.random.normal(0,level,len(dat))
    for j in range(len(local_noise)) :
        noise[position+j] += local_noise[j]
    dat = dat + noise


    drop = find_drop(dat, 0.3, False, False)
    assert np.abs(len_plateau-drop) <= 20






@given(intensity = st.integers(min_value=0, max_value=200))
def test_intensityi_in_image(intensity):
    """
    creates a tif image in a known path
    The image is 30x30 squares. the 30x10 middle line have values of 100 the others 1
    Averaging the ones in the middle the value should be 100

    GIVEN: a random walue of intensity
    WHEN: I measure the value of the pixels
    THEN: The result is the intensity
    """
    # Create an array representing the image
    image_array = np.zeros((30, 30), dtype=np.uint8)  # Initialize a 3x3 array with zeros
    image_array[10:20] = intensity  # Set the middle row to a random value



    # Save the array as a TIFF image
    imageio.imwrite('image.tif', image_array)

    assert intensity_in_image('image.tif',[0.35, 0.66], False) == intensity



def test_intensityi_in_multiple_image():
    """
    This test is to make sure that a multipage tif is read correctly by the function intensity_in_multiple_page()

    GIVEN: random values of pixels are given to the images of the multipage tif
    WHEN: measure the values of the intensity of the images
    THEN: the measured list has to be equal to the random one generated
    """
    I = []
    for l in range(100):
        I.append(np.random.randint(0, 200))
    
    # Create a list of images to be written to the multi-page TIFF
    image_list = []
    for i in range(len(I)):
        im =  np.ones((10, 10), dtype=np.uint16)
        im[:]=I[i]
        image_list.append(im)
    

    # Save the list of images as a multi-page TIFF
    imageio.mimwrite('multipage_image.tif', image_list, format='tif')

    assert set(I) == set(intensity_in_multipage_image('multipage_image.tif',[0.35,0.66]))


@given(b = st.floats(min_value=0, max_value=100), a = st.floats(0.1,1))
def test_exp_fit(b,a):
        
    """
    create some data simulating a noisy exponential and fit the values of a and b to verify they are well fit

    GIVEN: two random floats b in (0,100) and a in (0.1,1)
    WHEN: fitting the stretched exponential
    THEN: the fit mus give correct results
    """
    def st_exp(x,b,a):
        return np.exp(-(b*x)**a)

    #create exponential like array with a plateau before
    len_drop = 1000 # number of points of the drop
    len_plateau = 300 # number of points of the plateu
    fps = 300
    plateau = np.ones(len_plateau)
    t = np.arange(0,5,1/fps)
    exp = np.exp(-(6*t)**0.5)
    dat = np.append(plateau,exp)

    # add noise 
    level = 0.05
    noise = np.random.normal(0, level, len(dat))
    dat = dat + noise

    ret =  perform_st_exp_fit(t, dat, 300, len_drop, st_exp, False, fps, 'title')
    d = ret[0]
    alpha = ret[1]
    
    assert np.abs(d - 1) < THRESHOLD
    assert np.abs(alpha - 0.5) < THRESHOLD
