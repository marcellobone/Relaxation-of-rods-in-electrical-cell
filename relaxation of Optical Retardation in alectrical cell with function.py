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


def st_exponential(x, b, alpha): # stretched exponential
     return a_0 * np.exp( -(b * x)**alpha) + c_0 # a_0 and c_0 are defined later in the code


# %%

########### DATASET Specify the file path

par_pol_path = 'C:\\Users\\marc3\\OneDrive\\Documents\\INTERNSHIP-PHD\\4-16-24 MB03x100 ph4.5\\img0.tif'  # replace with the actual file path

## The 
# Specify the folder path where the files are located
folder_path = 'C:\\Users\\marc3\\OneDrive\\Documents\\INTERNSHIP-PHD\\4-16-24 MB03x100 ph4.5\\5sec 5V 1MHz\\example data biref'  # Update with the actual path

# Define the pattern for file names 
file_pattern = 'relax*.tif'        

# Create the full path pattern
full_path_pattern = os.path.join(folder_path, file_pattern)

# Use glob to find all files that match the pattern
file_paths = glob.glob(full_path_pattern)


############ PARAMETERS and settings
### GENERAL (also parallel pol) #########
show_patch           = True   #plugged in the intensity_in_image function
visualize_derivative = True   #plugged in the find_drop function
visualize_drop       = True   #plugged in the find_drop function
visualize_dataset    = False   #to visualize the data nd verify the automatic measure worked
visualize_fit        = True   #pluggd in the perform_exp_fit function
height_delimeters = [.2, .6]
I_0 = 10*intensity_in_image(par_pol_path, height_delimeters, show_patch)
time_of_recording = 5 #in seconds s

# %%
n = 0
D = []
alpha = []
err_D = []
err_alpha = []
num = 0

## this cycle is finding the drop and fitting it for every file found 
for file_path in file_paths:
    
    print('examining ',file_path, f'\n{num} {num} {num} {num} {num} {num} {num} {num} {num}')
    
    ### SPECIFIC TO SINGLE RELAXATION #######    
    I = intensity_in_multipage_image(file_path, height_delimeters)
    frames = np.arange(len(I))
    fps = len(I)/5

    #visualize dataset to verify it's good
    if visualize_dataset == True:    
        plt.plot(frames,I,marker = '.')
        plt.title('DATASET')
        plt.show()

    # change frames to time and I to optical retardation
    t = frames/fps
    OR = np.arcsin(np.sqrt(I/I_0)) # if you want to keep I just comment this
    OR = OR / np.mean(OR[1:50]) # to normalize OR !NOTE it can be too long for little fps!
    #OR = I


    begin_drop = find_drop(I,visualize_derivative,visualize_drop) # pos in the array at which the drop starts
    print('drop begins at ',begin_drop ,' or ',begin_drop/fps,'s')

    fit_len = int(0.6 * (len(OR) - begin_drop ))  #lenght of the fit
    b_0 = 10
    c_0 = np.mean(OR[begin_drop+fit_len-25 : begin_drop+fit_len] )  #OR[len(OR)-1] OR[begin_drop+fit_len]  max(OR)
    a_0 = np.mean(OR[1:begin_drop] ) - c_0

    ##############################################################


    
    ######### PERFORM A STRETCHED EXPONENTIAL FIT
    title = os.path.splitext(os.path.basename(file_path)) #is used in the function
    title = title [0]
    savepath =  os.path.join(folder_path, title)
    [d, a, err_d, err_a] = perform_st_exp_fit(t,OR,begin_drop,fit_len,st_exponential,visualize_fit,fps,title) # d and a are D and alpha
    D.append(d)
    alpha.append(a)
    err_D.append(err_d)
    err_alpha.append(err_a)
    
    num += 1
    print('\n')
    

print('D values are :', D)
print(r'$\alpha$ values are :', alpha)


# plotting the final results  NOTE we use the fps of the last dataset because they should be similar
for i in range(len(D)) :
    t = (1/fps)*np.arange(500)
    st_exp = np.exp( -(6* (D[i]*t))**alpha[i])
    plt.plot(t,st_exp, alpha = 0.8 , label = rf' relax {[i]}')
    
plt.xlabel('time [s]')
plt.ylabel('normalized OR')
plt.grid('fine')
plt.legend()
plt.show()


## * calculate the mean values of d and alpha and plot them 
D = np.array(D)
err_D = np.array(err_D)
alpha = np.array (alpha)
err_alpha = np.array(err_alpha)

D_avg, err_D_avg =  weighted_avg_and_std(D, err_D)

alpha_avg, err_alpha_avg = weighted_avg_and_std(alpha, err_alpha)

title_Da = 'D and alpha'
title_Da = os.path.join(folder_path,title_Da)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for i in range(len(D)):
    ax1.errorbar(i+1,D[i], yerr=err_D[i], fmt='-', color = 'b' ,marker = '$D$')
    ax2.errorbar(i+1,alpha[i], yerr=err_alpha[i], fmt='-', color = 'r', marker = r'$\alpha$')
ax1.axhline(D_avg, color = 'b', label = rf'D = {D_avg}')
ax1.axhline(D_avg+err_D_avg/2, color = 'b', linestyle = '--')
ax1.axhline(D_avg-err_D_avg/2, color = 'b', linestyle = '--')
ax2.axhline(alpha_avg, color = 'r', label = rf'$\alpha$ = {alpha_avg}')
ax2.axhline(alpha_avg+err_alpha_avg/2, color = 'r', linestyle = '--')
ax2.axhline(alpha_avg-err_alpha_avg/2, color = 'r', linestyle = '--')
ax2.set_ylim(0.5*min(alpha),1.1*max(alpha)) 
ax2.set_ylim(0.9*min(alpha),1.3*max(alpha))
ax1.set_ylabel(r'rotational diffusion coefficient D [s$^{-1}$]')
ax2.set_ylabel(r'exponent $\alpha$')
plt.grid()
ax1.legend(loc = 'upper right')
ax2.legend(loc = 'lower right')
plt.savefig(title_Da)
plt.show()

