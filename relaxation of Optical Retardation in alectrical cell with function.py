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

def st_exponential(x, b, alpha): # stretched exponential
     return a_0 * np.exp( -(b * x)**alpha) + c_0 # a_0 and c_0 are defined later in the code


# %%

TXT_MODE = False # if True skips the reading tifs and rreads the txt files insteaad


########### DATASET Specify the file path
# Specify the folder path where the files are located
folder_path = filedialog.askdirectory() #'C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/5-28-24 EG in oe cell'#filedialog.askdirectory()  
image_par_pol_filename = 'img0.tif'
par_pol_path = os.path.join(folder_path,image_par_pol_filename)  
dark_image_path = os.path.join(folder_path,'img_dark0.tif')

if TXT_MODE == False:
    # Define the pattern for file names 
    file_pattern = 'relax*.tif'        
if TXT_MODE == True:
    # Define the pattern for file names 
    file_pattern = 'relax*.txt'  
# Create the full path pattern
full_path_pattern = os.path.join(folder_path, file_pattern)

# Use glob to find all files that match the pattern
file_paths = glob.glob(full_path_pattern)


############ PARAMETERS and settings
### GENERAL (also parallel pol) #########
show_patch           = True   #plugged in the intensity_in_image function
visualize_derivative = False   #plugged in the find_drop function
visualize_drop       = False   #plugged in the find_drop function
visualize_dataset    = True   #to visualize the data nd verify the automatic measure worked
visualize_fit        = True   #pluggd in the perform_exp_fit function
dark_image           = False
height_delimeters = [.45, .65]
I_0 = 10000*intensity_in_image(par_pol_path, height_delimeters, show_patch)
time_of_recording = 22 #in seconds s
#PARAMETERS for find_drop: (see function description)
threshold = 0.3 #threshold in percentage to find drop higher for noisy data


# %%
n = 0
D = []
alpha = []
err_D = []
err_alpha = []
num = 0

#%% EG
dim = 2000
I_EG = (4146.027685950413-3621.4595041322314)* np.exp( -(6*0.139826* np.arange(dim)/73.8)**0.530546) + 3621.4595041322314 
#%%


## this cycle is finding the drop and fitting it for every file found 
for file_path in file_paths:
    
    print('examining ',file_path, f'\n{num} {num} {num} {num} {num} {num} {num} {num} {num}')
    
    ### SPECIFIC TO SINGLE RELAXATION #######   
    if TXT_MODE == True :
        I = np.loadtxt(file_path,  delimiter='\t', usecols=(0), unpack=True)
    if TXT_MODE == False:
        I = intensity_in_multipage_image(file_path, height_delimeters)
    
    frames = np.arange(len(I))
    fps = len(I)/time_of_recording

    #visualize dataset to verify it's good
    if visualize_dataset == True:    
        plt.plot(frames,I,marker = '.')
        plt.title('DATASET')
        plt.show()

    begin_drop = find_drop(I,threshold,visualize_derivative,visualize_drop) # pos in the array at which the drop starts
    print('drop begins at ',begin_drop ,' or ',begin_drop/fps,'s')


    # change frames to time and I to optical retardation
    t = frames/fps
    OR = np.arcsin(np.sqrt(I/I_0)) # if you want to keep I just comment this
    # correction_EG = 0.001*np.exp(-(6*0.2*t[0:len(OR)-begin_drop-1]) **0.8)
    # plt.plot(OR/np.mean(OR[1:10]))
    # OR[begin_drop:len(OR)] = OR[begin_drop:len(OR)] - correction_EG
    # OR[0:begin_drop-1] = OR[0:begin_drop-1]-0.001
    OR = OR / np.mean(OR[1:10]) # to normalize OR !NOTE it can be too long for little fps!
    
   


    fit_len = int(0.6* (len(OR) - begin_drop ))  #lenght of the fit
    b_0 = 10
    if dark_image == False:
        c_0 = min(OR)#np.mean(OR[len(OR)-10:len(OR)] )  #OR[len(OR)-1] OR[begin_drop+fit_len]  max(OR)
    if dark_image == True:
        I_dark = intensity_in_image(dark_image_path, height_delimeters, False)
        c_0 = np.arcsin(np.sqrt(I_dark/I_0))
        c_0 = c_0 / np.mean(OR[1:10])
        plt.plot(I)
        plt.axhline(I_dark)
        plt.show()
    a_0 = np.mean(OR[begin_drop-3:begin_drop] ) - c_0


    ##############################################################


    
    ######### PERFORM A STRETCHED EXPONENTIAL FIT
    title = os.path.splitext(os.path.basename(file_path)) #is used in the function
    title = title [0]
    savepath =  os.path.join(folder_path, title)
    [d, a, err_d, err_a] = perform_st_exp_fit(t,OR,begin_drop,fit_len,st_exponential,visualize_fit,fps,title,savepath) # d and a are D and alpha
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
    t = (1/fps)*np.arange(1000)
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
ax1.set_ylim(0.5*min(D),1.1*max(D)) 
ax2.set_ylim(0.9*min(alpha),1)
ax1.set_ylabel(r'rotational diffusion coefficient D [s$^{-1}$]')
ax2.set_ylabel(r'exponent $\alpha$')
plt.grid()
ax1.legend(loc = 'upper right')
ax2.legend(loc = 'lower right')
plt.savefig(title_Da)
plt.show()



# %%
minEG = 3646.9024
maxEG = 4154.0588

I_EG =   (maxEG-minEG)* np.exp( -(6*0.1346* np.arange(len(frames)-begin_drop)/fps)**0.5347) + minEG 
OR_EG =  np.arcsin(np.sqrt(I_EG/219595.71264367815))
OR_EG = OR_EG/max(OR_EG)

plt.plot(OR[begin_drop:len(frames)]-min(OR), label = 'I')
plt.plot(OR_EG-min(OR_EG), label = 'EG')
plt.plot()
plt.legend()
plt.show()
# %% create file with useful things
resultfilename = 'results.txt'
resultfile_path = os.path.join(folder_path,resultfilename)
with open(resultfile_path,'w') as file:
    file.write('D errD alpha erralpha maxI minI \n')
    file.write(f'{D_avg} {err_D_avg} {alpha_avg} {err_alpha_avg} {max(OR)} {min(OR)} ')
    file.close()

TEB_path = os.path.join(folder_path,f'TEBdata_fps_{fps}_.txt')
with open(TEB_path, 'w') as file:
        for value in OR[begin_drop:]:
         file.write(f"{value}\n")
    

# %%
