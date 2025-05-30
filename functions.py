import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.widgets import Slider
########### FUNCTIONS 
def find_drop(I,threshold_perc,plot_deriv,plot_begin) : # finds the beginning of the drop of the exponential-like dataset
    """
    the function first smoothens the data in order to reduce noise. 
    It does this by averaging every point with its first d neighbours.
    d is a fixed parameter in the function
    Then a threshold is set to be 10% of the maximum peak. This can need to be changed in
    very noisy datasets ot datasets with some aberration.
    At every peak lower than the threshold the function cheks if there is an 
    actual drop by veryfying that the the points 'dive' deep enough.
    Meaning the following procedure is used.
    the total drop is defined as max - min of the data. 
    the dive of a certain point is defined as:
    the value of the point - the value of the point at a distance of 10% of 
    the toral number of points in the dataset.
    If dive > 20%drop then we break  the cycle and return the index of said point 

    inputs :
     I --array a dataset of exponential like points 
     plot_deriv --boolean indicating if you want to plot the derivative
     plot_begin --boolean indicating if you want to plot the drop
     threshold_perc --float threshold in percentaage wrt the max derivaative
     dive_perc --float dive in perchentage wrt the drop
    Outputs:
     begin_drop --int index at which the drop begins
    
    """
    # smoothening data to have cleaner derivative
    I_smooth = []
    d = 5
    
    for i in range(d+1,len(I)-d-1) : 
        I_smooth.append(( np.mean(I[i-d+1:i+d+1])))
    
   
    #plt.plot(I,marker = '.')      
    #plt.plot(I_smooth)
    #plt.show()

    derivative = np.gradient(I_smooth)
    derivative = [ min(derivative[i],0) for i in range(len(derivative)) ] #remove positive values
   
    # find the begin of drop looking at the derivative

    begin_drop = 0

    threshold = -threshold_perc*max(np.abs(derivative)) 
    possible_peak = []
    dist = int( 0.015*len(derivative) )
    for d in range(10,len(derivative)-dist) :
        if (derivative[d]) < threshold :
            possible_peak.append(d)
    #print(possible_peak)
   
    for pp in possible_peak :
        if np.abs( I[pp]- I_smooth[pp+dist] ) > np.abs( I[begin_drop] - I_smooth[begin_drop+dist] ) : 
            begin_drop = pp
            
            
            #dive = I_smooth[d]-I_smooth[d+int(0.1*len(I_smooth))]
            #if  dive > dive_perc*drop :
             #   begin_drop = d
              #  break
    
    
    #print('drop begins at ',begin_drop ,' or ',begin_drop/fps,'s')

    if plot_deriv == True :
        plt.plot(derivative , marker = '.')
        plt.axhline(threshold, color = 'r' )
        plt.title('derivative')
        plt.grid('minor')
        plt.show()
    
    if plot_begin == True :
        frames = np.arange(1,len(I)+1)
        plt.plot(frames, I, color = 'b', marker = '.')
        plt.plot(frames[begin_drop : len(frames)] , I[begin_drop : len(I)], color = 'r')
        plt.show()
    return(begin_drop)

def intensity_in_image(path, height_delimeter, show_patch) : # calculate the intensity in an tif image inside a ROI

    """
    It calculates the average values of the pixel inside a region of interest of a tif image.
    Keep in mind that the borders of the region of interest are rounded down to an int
    so it is better to give stricter values for the uper limit of the ROI

    Inputs:
     path --string path to the image
     height_delimeters --float percentaage (in 0 to 1) of heigth at which to place the upper and lower bounds
              of the ROI. NOTE that the yaxis is reversed and the fist number should be the lowest
     show_patch --boolean if true plots the ROI over the image  

    Output:
     I_0 --float vaalue of the average pixel in the ROI       
    """

    with Image.open(path) as par_pol :
        # Convert the Pillow image to a NumPy array (OpenCV format)
        par_pol_np = np.array(par_pol)
        width, height = par_pol.size
        width-=1
        height-=1
        #print(f'w = {width}   h = {height}')
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

def intensity_mat(path):
    with Image.open(path) as par_pol :
    
        # Convert the ROI to a NumPy array for intensity analysis
        I = np.array(par_pol)

        return(I) 


def intensity_in_multipage_image(path,height_delimeter) : # finds the average intensity inside a ROI of a multipage tif and makes a array with the value
    
    """
    Inputs are a string containing the path to the image and 
    a list of two numbers determining the height limits of the region of interest
    in percentage with respect to the total height.
    It returns the average value of the pixel inside the ROI

    Inputs: 
     path --string with path to the multi page tif
     heigth_delimeters --float percentaage (in 0 to 1) of heigth at which to place the upper and lower bounds
              of the ROI. NOTE that the yaxis is reversed and the fist number should be the lowest
    Output:
     I --arrray contaaining the average value of the pixels in the ROI of per every page in the multi page tif.
    """

    image = Image.open(path)

    image.seek(0)
    width, height = image.size

    # Define the region of interest (ROI) coordinates (left, upper, right, lower)
    roi_coordinates = (0, int(height_delimeter[0]*height) , width , int(height_delimeter[1]*height ))

    # length of the TIF
    num_pages = image.n_frames

    I = []
    for page in range(num_pages):
        image.seek(page)
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

def perform_st_exp_fit(t,OR,begin_drop,fit_len, st_exponential, plot_fit, fps, title,savepath): # performs a fit with the stretched exponential function
    """
    Performs a exponential fit of form exp(-(bt)**alpha)

    Inputs:
        t --array with times
        OR --array with a data
        begin_drop --int indicating the index at which the exponential drop begins  
        fit_len --int indicating how many points to fit starting from the drop
        st_exponential --stretched exponential function
        plot_fit --boolean True for plotting }
        fps --float fps of the recording     } only relevant if plotting
        title --string title  of plot        }
        savepath --string path to save fig   }

    Outputs fitted values:
     list containing:
        D --float with value b/6
        err_D --float error on D
        alpha --float
        err_alpha --float error on alpha

    """
    # Subset of the data for fitting
    t_fit = t[begin_drop : begin_drop + fit_len]-t[begin_drop]
    OR_fit = OR[begin_drop : begin_drop + fit_len] #- OR[len(OR)-1]

    initial_guesses = [17, 0.2]
    lower_bounds = [-np.inf, -np.inf]
    upper_bounds = [np.inf, 1]
    # Perform the exponential fit on the subset
    params, covariance = curve_fit(st_exponential, t_fit, OR_fit,maxfev = 50000, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))

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

    
    
    if plot_fit == True:
        D = f"{b_fit/6:.3f}"
        plt.plot(t[begin_drop-int(begin_drop/2):begin_drop+fit_len+int((len(t)-fit_len)/2)],OR[begin_drop-int(begin_drop/2):begin_drop+fit_len+int((len(t)-fit_len)/2)], marker = '.', linestyle = 'none')
        
        # Plot error bands around the fitted curve for b_fit and alpha_fit
        #num_curves = 100
        #perturbed_params = np.random.multivariate_normal(params, covariance, num_curves)
        #for i in range(num_curves):
         #   perturbed_curve = st_exponential(t_fit, *perturbed_params[i])
          #  plt.plot(t_fit + (begin_drop+1)/fps, perturbed_curve, color='orange')

        plt.plot(t_fit + (begin_drop)/fps, fit_curve, color='red', label=rf' D={D}  $\alpha$={f"{alpha_fit:.3f}"} ')
        plt.xlabel('time [s]')
        plt.ylabel('optical retardation')
        plt.legend(loc = 'best')
        plt.grid()
        plt.title(title)
        plt.savefig(savepath)
        plt.show()
        

    return([b_fit/6, alpha_fit, err_b/6, err_alpha])

def weighted_avg_and_std(values, weights): #perform weighted average
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))




def fit_dist(bins,hist_values, degree):
    
    
    cum_dist = np.cumsum(hist_values)

    # Generate corresponding x values (assuming equally spaced)
    x = np.arange(len(cum_dist))

    # Fit a polynomial of the specified degree to the data
    coefficients = np.polyfit(x, cum_dist, degree)

    # Create a polynomial object
    polynomial = np.poly1d(coefficients)

    # Generate x values for plotting the fit
    x_fit = np.linspace(x.min(), x.max(), 1000)
    fit_cum_dist = polynomial(x_fit)

    # Plot the original data
    plt.scatter(x, cum_dist, label='Data')

    # Plot the polynomial fit
    plt.plot(x_fit, fit_cum_dist, label=f'Polynomial Fit (degree {degree})', color='red')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial Fit to Data')
    plt.show()

    dist = np.diff(fit_cum_dist)

    plt.hist(hist_values,len(bins), density = True)
    plt.plot(x,dist*max(hist_values)/max(dist))
    plt.show()
    return(dist)
    

def interactive_fit(x_data, y_data, x_offset, a0, D0, alpha0, c0):
    """
    Create an interactive plot with sliders to adjust parameters of the model function.
    """
    # Define the model function
    def st_exponential(x, a,D,c,alpha): # stretched exponential
         return a * np.exp( -((6*D) * x)**alpha) + c
        


    # Create the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Plot dataset
    scatter = ax.scatter(x_data-x_offset, y_data, label='Data', color='gray')

    # Plot initial model
    model_line, = ax.plot(x_data, st_exponential(x_data, a0, D0, c0, alpha0), label='Model', color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    # Define sliders for parameters
    a_slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    D_slider_ax = fig.add_axes([0.25, 0.20, 0.65, 0.03])
    c_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    alpha_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03])

    a_slider = Slider(a_slider_ax, 'a', 0.0001, 0.05, valinit=a0)
    D_slider = Slider(D_slider_ax, 'D', 0.1, 50, valinit=D0)
    c_slider = Slider(c_slider_ax, 'c', 0, 0.05, valinit=c0)
    alpha_slider = Slider(alpha_slider_ax, 'alpha', 0.01, 1, valinit=a0)
    # Update function for sliders
    def update(val):
        a = a_slider.val
        D = D_slider.val
        c = c_slider.val
        alpha = alpha_slider.val

        model_line.set_ydata(st_exponential(x_data, a, D, c, alpha))
        fig.canvas.draw_idle()

    # Connect sliders to update function
    a_slider.on_changed(update)
    D_slider.on_changed(update)
    c_slider.on_changed(update)
    alpha_slider.on_changed(update)

    plt.xlim(0, max(  x_data - x_offset))

    plt.show()

    print('a = ', a_slider.val)
    print('D = ', D_slider.val) 
    print('c = ', c_slider.val)
    print('alpha = ', alpha_slider.val)
    return a_slider.val, D_slider.val, c_slider.val, alpha_slider.val
