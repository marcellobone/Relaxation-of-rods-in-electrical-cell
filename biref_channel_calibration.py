# %%
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
from functions import *
from tkinter import filedialog
import re

def extract_numbers_from_filenames(directory):
    # Regular expression to match the pattern 'flow x nlmin.tif'
    pattern = re.compile(r'flow (\d+) nlmin\.tif')

    # List to store extracted numbers
    numbers = []

    # Iterate through files in the specified directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the number and convert it to an integer
            number = int(match.group(1))
            numbers.append(number)

    return numbers


########### DATASET Specify the file path
# Specify the folder path where the files are located
folder_path = filedialog.askdirectory() #'C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/5-28-24 EG in oe cell'#filedialog.askdirectory()  
image_par_pol_filename = 'img0.tif'
par_pol_path = os.path.join(folder_path,image_par_pol_filename)  
flow = extract_numbers_from_filenames(folder_path)


# Define the pattern for file names 
file_pattern = 'flow*.tif'        

# Create the full path pattern
full_path_pattern = os.path.join(folder_path, file_pattern)

# Use glob to find all files that match the pattern
file_paths = glob.glob(full_path_pattern)
I_0 = intensity_in_image(par_pol_path,[0.2,0.8],True)
OR_all = []
avg_OR = []
OR_cross_sec_all = []
i = 0
for file_path in file_paths:
    print('examining ----', file_path, '\n')
    I = intensity_mat(file_path)
    print(max(I[28]))
    if flow[file_paths.index(file_path)] < 2000: 
        OR = np.arcsin((np.sqrt(I/(200*I_0))))
        print('true')
    else :
        OR = np.arcsin((np.sqrt(I/(100*I_0))))
        print('false')
    OR_cross_sec = []
    for i in range(len(OR[10][:])):
        OR_cross_sec.append(np.mean(OR[:][i])/50)
    OR_all.append(OR_cross_sec)
    plt.plot(OR_cross_sec, label = f'{flow[file_paths.index(file_path)]}' )
    avg_OR.append(np.mean(OR_cross_sec[15:30]))
    i = i+ 1
plt.xlabel(fr'channel cross section [$\mu$m]')
plt.ylabel('birefringence')
#plt.legend()
plt.show()
biref = [avor/50 for avor in avg_OR]
#plt.plot(flow,biref, '.')
#plt.ylabel('birefringence')
#plt.xlabel('flow [nl\min]')

# %%

import math

def convert_flow_rate_to_m3_per_s(flow_rate_nl_per_min):
    # Convert nanoliters per minute to cubic meters per second
    flow_rate_m3_per_s = flow_rate_nl_per_min * 1e-9 / 60
    return flow_rate_m3_per_s

def calculate_velocity(flow_rate, shape, dimensions):
    if shape == 'rectangular':
        w, h = dimensions
        area = w * h
    elif shape == 'circular':
        d, = dimensions
        area = math.pi * (d / 2) ** 2
    elif shape == 'trapezoidal':
        b, t, h = dimensions
        area = ((b + t) / 2) * h
    elif shape == 'elliptical':
        a, b = dimensions
        area = math.pi * a * b
    else:
        raise ValueError("Unsupported shape")
    
    # Calculate the velocity
    velocity = flow_rate / area
    return velocity


def calculate_flow_rate(shape, dimensions, velocity):
    if shape == 'rectangular':
        w, h = dimensions
        area = w * h
    elif shape == 'circular':
        d, = dimensions
        area = math.pi * (d / 2) ** 2
    elif shape == 'trapezoidal':
        b, t, h = dimensions
        area = ((b + t) / 2) * h
    elif shape == 'elliptical':
        a, b = dimensions
        area = math.pi * a * b
    else:
        raise ValueError("Unsupported shape")

    flow_rate = area * velocity
    return flow_rate

# Convert the flow rate to cubic meters per second
flow_rate_m3_per_s = (convert_flow_rate_to_m3_per_s(f) for f in flow)

# Define the shape and dimensions of the channel
shape = 'rectangular'
dimensions = (0.00005, 0.00005)  # width = 0.02 meters, height = 0.01 meters

# Calculate the average velocity
av_velocity = []
for f in flow_rate_m3_per_s:
    av_velocity.append(calculate_velocity(f, shape, dimensions))

print(av_velocity)


import numpy as np
from scipy.integrate import dblquad

# Define constants

a = 50  # Half-width of the square channel

# Define the integrand
max_biref = 0.13
average_velocity_gradient = []
# Perform the double integral
for av in av_velocity:
    def integrand(y, x, av, a):
        return 2 * av * np.sqrt(x**2 + y**2) / a**2
    result, error = dblquad(integrand, -a, a, lambda x: -a, lambda x: a, args=(av, a))
    average_velocity_gradient.append(result)
# Calculate the average velocity gradient


# %%
f = [0,0.0165,0.0329,0.0492,0.0651,0.0807,0.0958,0.1105,0.1246,0.1382,0.1512,0.1756,0.1977,0.2177,0.2359,0.2524,0.2812,0.3054,0.3262,0.3443,0.3603,0.3933,0.4193,0.4406,0.4585,0.4737,0.4878,0.5113,0.5308,0.5478,0.5626,0.5763,0.5991,0.6314,0.6511,0.685]
alpha = [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12.5,15,17.5,20,22.5,25,30,35,40,45,50,60,80,100,200]
plt.plot(alpha,f,label= 'calib')
average_velocity_gradient = np.array(average_velocity_gradient)
#v_gradient = np.array(flow_rate_m3_per_s)*6/(50e-6)**3 
biref = np.array(biref)
plt.plot(average_velocity_gradient/40,biref*24000, '.')
plt.xlabel(r'$\dot{\gamma}/D$')
plt.ylabel('f')
plt.show()
# %%
v_gradient = (f*6/(50e-6)**3 for f in flow_rate_m3_per_s)
# %%
