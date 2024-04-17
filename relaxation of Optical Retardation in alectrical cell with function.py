import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob

########### FUNCTIONS 
def find_drop(I,plot_deriv,plot_begin) :
        
    # smoothening data to have cleaner derivative
    I_smooth = []
    d = 3
    for i in range(d,len(I)-d) : 
        I_smooth.append(( np.mean(I[i-d:i+d])))
    

    #plt.plot(frames,I,marker = '.')      ## uncomment to verify smoothening went well
    #plt.plot(I_smooth)
    #plt.show()


    derivative = np.gradient(I_smooth)
    derivative = derivative**3  # increase the high value and decrease the low ones
    
    derivative = [ min(derivative[i],0) for i in range(len(derivative)) ] # filter out the positive values
   
    # find the begin of drop looking at the derivative
    begin_drop = 0 
    threshold = -0.1*max(np.abs(derivative[5:int( 0.5*len(derivative) )])) # the 5 and 30 are there because it happens that there are weird artifacts at the first and last frames
   
    for d in range(10,int(len(derivative)/2)):
        if (derivative[d]) < threshold :
            dive = I_smooth[d]-I_smooth[d+int(0.1*len(I_smooth))]
            print(d,dive/(0.2*drop))
            if  dive > 0.2*drop :
                begin_drop = d
                break
    
    # print('drop begins at ',begin_drop ,' or ',begin_drop/fps,'s')
    if plot_deriv == True :
        plt.plot(derivative , marker = '.')
        plt.title('derivative')
        plt.grid('minor')
        plt.show()

    if plot_begin == True :
        plt.plot(frames, I, color = 'b', marker = '.')
        plt.plot(frames[begin_drop : len(frames)] , I[begin_drop : len(I)], color = 'r')
        plt.show()
    return(begin_drop)

def st_exponential(x, b, alpha):
    return a_0 * np.exp( -(b * x)**alpha) + c_0

def intensity_in_image(path, height_delimeter, show_patch) :

    with Image.open(path) as par_pol :
        # Convert the Pillow image to a NumPy array (OpenCV format)
        par_pol_np = np.array(par_pol)
        width, height = par_pol.size

        print(f'w = {width}   h = {height}')
        # Define the region of interest (ROI) coordinates (left, upper, right, lower)
        roi_coordinates = (0, int(height_delimeter[0]*height) , width , int(height_delimeter[1]*height ))
        
        if show_patch == True:
            # Create a subplot and show the original image
            fig, ax = plt.subplots()
            ax.imshow(par_pol, cmap='gray')

            # Create a rectangle patch representing the ROI
            roi_patch = patches.Rectangle((roi_coordinates[0], roi_coordinates[1]),
                                        roi_coordinates[2], roi_coordinates[3]-roi_coordinates[1],
                                        linewidth=2, edgecolor='r', facecolor='none')

            # Add the rectangle patch to the plot
            ax.add_patch(roi_patch)
            
            plt.title('ROI')

            # Show the plot
            plt.show()

        # Crop the image to the specified ROI
        roi = par_pol.crop(roi_coordinates)

        # Convert the ROI to a NumPy array for intensity analysis
        roi_array = np.array(roi)

        # Calculate the average intensity of the ROI
        I_0 = np.mean(roi_array)

        return(I_0) 

def intensity_in_multipage_image(path,height_delimeter) :
    
    image = Image.open(path)

    image.seek(0)
    width, height = image.size

    # Define the region of interest (ROI) coordinates (left, upper, right, lower)
    roi_coordinates = (0, int(height_delimeter[0]*height) , width , int(height_delimeter[1]*height ))

    # length of the TIF
    num_pages = image.n_frames

    I = []
    for page in range(num_pages):
        selected_page = image.seek(page)
        selected_page_image = image.copy()

        # Convert the Pillow image to a NumPy array (OpenCV format)
        image_np =  np.array(selected_page_image)

        # Extract the ROI from the image
        roi = image_np[roi_coordinates[1]:roi_coordinates[1] + roi_coordinates[3],
                    roi_coordinates[0]:roi_coordinates[0] + roi_coordinates[2]]

        # Measure intensity (you can use various methods depending on your requirement)
        average_intensity = cv2.mean(roi)[0]

        I.append(average_intensity)

       
    return(I)

def perform_st_exp_fit(t,OR,begin_drop,fit_len,plot_fit):

    # Subset of the data for fitting
    t_fit = t[begin_drop : begin_drop + fit_len]-t[begin_drop]
    OR_fit = OR[begin_drop : begin_drop + fit_len] #- OR[len(OR)-1]

    initial_guesses = [b_0, 20]

    # Perform the exponential fit on the subset
    params, covariance = curve_fit(st_exponential, t_fit, OR_fit,maxfev = 50000, p0=initial_guesses)

    # Extract the fitted parameters
    b_fit, alpha_fit = params
    var_b, var_alpha = np.diag(covariance)
    err_b = np.sqrt(var_b)
    err_alpha = np.sqrt(var_alpha)

    # Generate the fitted curve using the parameters
    fit_curve = st_exponential(t_fit, b_fit, alpha_fit)

    # Print the fitted parameters
    print("Fitted Parameters:")
    print("alpha =", alpha_fit)
    print("b =", b_fit)


    print( "OR ----- if b = 6D then D = ", b_fit/6 ,"s^-1")
    #print( "I  ----- if b/2 = 6D then D = ", b_fit/12 ,"s^-1")

    
    D = f"{b_fit/6:.3f}"
    if plot_fit == True:

        plt.plot(t[begin_drop-100:begin_drop+fit_len+150],OR[begin_drop-100:begin_drop+fit_len+150], marker = '.', linestyle = 'none')
        
        # Plot error bands around the fitted curve for b_fit and alpha_fit
        num_curves = 100
        perturbed_params = np.random.multivariate_normal(params, covariance, num_curves)
        for i in range(num_curves):
            perturbed_curve = st_exponential(t_fit, *perturbed_params[i])
            plt.plot(t_fit + (begin_drop+1)/fps, perturbed_curve, color='orange')

        plt.plot(t_fit + (begin_drop+1)/fps, fit_curve, color='red', label=rf' D={D}  $\alpha$={f"{alpha_fit:.3f}"} ')
        
        plt.xlabel('time [s]')
        plt.ylabel('optical retardation')
        plt.legend(loc = 'best')
        plt.grid()
        plt.title(title)
        plt.savefig(savepath)
        plt.show()

    return([b_fit/6, alpha_fit, err_b/6, err_alpha])

########### DATASET Specify the file path

par_pol_path = 'C:\\Users\\marc3\\OneDrive\\Documents\\INTERNSHIP-PHD\\3-21-24 MB03x100\\img0.tif'  # replace with the actual file path

## The 
# Specify the folder path where the files are located
folder_path = 'C:\\Users\\marc3\\OneDrive\\Documents\\INTERNSHIP-PHD\\3-21-24 MB03x100'  # Update with the actual path

# Define the pattern for file names 
file_pattern = 'relax*.tif'        

# Create the full path pattern
full_path_pattern = os.path.join(folder_path, file_pattern)

# Use glob to find all files that match the pattern
file_paths = glob.glob(full_path_pattern)


############ PARAMETERS and settings
### GENERAL (also parallel pol) #########
show_patch           = False   #plugged in the intensity_in_image function
visualize_derivative = False   #plugged in the find_drop function
visualize_drop       = False   #plugged in the find_drop function
visualize_dataset    = False   #to visualize the data nd verify the automatic measure worked
visualize_fit        = True   #pluggd in the perform_exp_fit function
height_delimeters = [.4, .6]
I_0 = intensity_in_image(par_pol_path, height_delimeters, show_patch)

n = 0
D = []
alpha = []
err_D = []
err_alpha = []
num = 0
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
    [d, a, err_d, err_a] = perform_st_exp_fit(t,OR,begin_drop,fit_len,visualize_fit) # d and a are D and alpha
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

variance_D = np.var(D)
covariance_matrix_D = np.cov(D, rowvar=False)

D_avg = np.sum(D * err_D) / np.sum(err_D)
err_D_avg = np.sqrt(np.sum((err_D ** 2) * variance_D) + 2 * np.dot(err_D, np.dot(covariance_matrix_D, err_D)))

variance_alpha = np.var(alpha)
covariance_matrix_alpha = np.cov(alpha, rowvar=False)

alpha_avg = np.sum(alpha * err_alpha) / np.sum(err_alpha)
err_alpha_avg = np.sqrt(np.sum((err_alpha ** 2) * variance_alpha) + 2 * np.dot(err_alpha, np.dot(covariance_matrix_alpha, err_alpha)))

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

