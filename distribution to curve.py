# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
from functions import *
from scipy.stats import skewnorm
from ipywidgets import interactive
from tkinter import filedialog
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk


def lognorm_pdf(x, sig, LM):
    return (1/(x*sig*np.sqrt(2*np.pi)))*np.exp( -np.log(x/LM)**2/(2*sig**2) )

def st_exponential(x, b, alpha): # stretched exponential
     return np.exp( -(b * x)**alpha) # a_0 and c_0 are defined later in the code

def SE_from_TEBcurve(time,curve):
    if curve[0] != 1:
        curve = curve/curve[0]
    params, covariance = curve_fit(st_exponential,time,curve)
       # Extract the fitted parameters
    b_fit, alpha_fit = params
    var_b, var_alpha = np.diag(covariance)
    err_b = np.sqrt(var_b)
    err_alpha = np.sqrt(var_alpha)

    # Generate the fitted curve using the parameters
    fit_curve = st_exponential(time, b_fit, alpha_fit)
    
    return fit_curve, b_fit/6, alpha_fit

sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
L_median = 120
# %% CREATING DISTRIBUTION


# for sigma in sigmas :

#     lengths =  np.linspace(5,1000,1000)
#     ar = 15


#     dist = lognorm_pdf(lengths, sigma, L_median)
#     V = np.pi*( (0.5*lengths/ar)**2)*lengths
#     eff_dist = V*dist
#     eff_dist = eff_dist / sum(eff_dist)


#     #### CREATE CURVE

#     t = np.linspace(0,2,1000)

#     T = 300



#     def rot_diff_par(l,AR) :
        
#         eta = np.exp(-2.3568+617.2/(T-179.414))*10**-2   #2*10**(-1)	  
#         kT = 4.11*10**(-21)/300 * T
#         S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
#         A = (3*kT)/(np.pi*eta*l**3)
#         G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
#         D = A*G
#         return D


#     D_l = rot_diff_par(lengths*10**(-9),ar)

#     # effecctive relax
#     exps = []
#     for i in range(len(D_l)):
#         exps.append(np.exp(-6*D_l[i]*t) )

#     w_exps = []
#     for i in range(len(eff_dist)) :
#         w_exps.append(eff_dist[i]*exps[i])


#     relax = []
#     somma_exps = 0
#     for j in range(len(t)):
#         for i in range(len(D_l)):
#             somma_exps = somma_exps  + w_exps[i][j]
#         relax.append(somma_exps)
#         somma_exps = 0

#     relax_corrected_EG = relax + 0.1*np.exp(-(6*0.4*t)**0.8)
#     relax_corrected_EG = relax_corrected_EG/relax_corrected_EG[0]



#     ax1.plot(lengths,dist,  label = fr' $ \sigma $={sigma} $L_M$={L_median}' )
#     ax2.plot(t,relax_corrected_EG)
 


# ax2.set_xlabel('x')
# ax2.set_ylabel('')



# ax1.set_xlabel('x')
# ax1.set_ylabel('size dist')
# ax1.legend()

# plt.show()




# %%


# Function to get user input for the number of lines
def gret_sigma_value():
    while True:
        try:
            sigma = float(input("Enter SIGMA: "))
            if sigma > 0 :
                return sigma
            else:
                print("Please enter a value between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Function to get user input for the function type
def get_LM_type():
    while True:
        try:
            L_median = int(input("Enter L median: "))
            if L_median > 0 :
                return L_median
            else :
                print("Please enter a positive value.")
        except ValueError :
            print("BOOOOOh.")

def generate_data_and_plot(sigma, L_median):
    # Example data generation, replace with your actual data generation logic
    x = np.linspace(0, 10, 1000)
    y = np.exp(-(x - L_median)**2 / (2 * sigma**2))

    ax.clear()
    ax.plot(x, y, label=f'sigma={sigma}, L_median={L_median}')
    ax.set_title('Generated Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    canvas.draw()


def plot_dist_and_fit(L_median , sigma  ):
    fig, (ax1, ax2) = plt.subplots( 2,1)
    lengths =  np.linspace(5,1000,1000)
    ar = 15
    #sigma = gret_sigma_value()
    #L_median = get_LM_type()




    dist = lognorm_pdf(lengths, sigma, L_median)
    V = np.pi*( (0.5*lengths/ar)**2)*lengths
    eff_dist = V*dist
    eff_dist = eff_dist / sum(eff_dist)


    #### CREATE CURVE

    t = np.linspace(0,2,2000)
    T = 300

    def rot_diff_par(l,AR) :
        
        eta = np.exp(-2.3568+617.2/(T-179.414))*10**-2   #2*10**(-1)	  
        kT = 4.11*10**(-21)/300 * T
        S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
        A = (3*kT)/(np.pi*eta*l**3)
        G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
        D = A*G
        return D


    D_l = rot_diff_par(lengths*10**(-9),ar)

    # effecctive relax
    exps = []
    for i in range(len(D_l)):
        exps.append(np.exp(-6*D_l[i]*t) )

    w_exps = []
    for i in range(len(eff_dist)) :
        w_exps.append(eff_dist[i]*exps[i])


    relax = []
    somma_exps = 0
    for j in range(len(t)):
        for i in range(len(D_l)):
            somma_exps = somma_exps  + w_exps[i][j]
        relax.append(somma_exps)
        somma_exps = 0

    relax_corrected_EG = relax + 0.1*np.exp(-(6*0.4*t)**0.8)
    relax_corrected_EG = relax_corrected_EG/relax_corrected_EG[0]

    SE, D_eff, alpha = SE_from_TEBcurve(t,relax_corrected_EG)

    ax1.plot(lengths[0:500],dist[0:500],  label = fr' $ \sigma $={sigma} $L_M$={L_median}' )
    ax2.plot(t,relax_corrected_EG, label = fr'$\alpha$={alpha}, D_eff={D_eff}' )
    ax2.plot(t,SE,':')



    ax2.set_xlabel('x')
    ax2.set_ylabel('normalized biref')
    ax2.legend()


    ax1.set_xlabel('x')
    ax1.set_ylabel('size dist')
    ax1.legend()
    

    plt.show()


# Update plot function
def update_plot(event=None):
    sigma = sigma_var.get()
    L_median = L_median_var.get()
    generate_data_and_plot(sigma, L_median)

# Create the main window
root = tk.Tk()
root.title("Interactive Plot with Tkinter")

# Create matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create sliders for sigma and L_median
controls_frame = ttk.Frame(root)
controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

sigma_var = tk.DoubleVar(value=1.0)
L_median_var = tk.DoubleVar(value=5.0)

ttk.Label(controls_frame, text="Sigma:").grid(row=0, column=0)
sigma_slider = ttk.Scale(controls_frame, from_=0.1, to_=5.0, variable=sigma_var, orient=tk.HORIZONTAL, command=update_plot)
sigma_slider.grid(row=0, column=1, sticky='ew')

ttk.Label(controls_frame, text="L_median:").grid(row=1, column=0)
L_median_slider = ttk.Scale(controls_frame, from_=0.0, to_=10.0, variable=L_median_var, orient=tk.HORIZONTAL, command=update_plot)
L_median_slider.grid(row=1, column=1, sticky='ew')

# Set the initial plot
generate_data_and_plot(sigma_var.get(), L_median_var.get())

# Start the Tkinter main loop
root.mainloop()
""""
# Create interactive widgets
LM_slider = widgets.FloatSlider(value=200, min=5, max=500, step=5, description='L median:')
sigma_slider = widgets.FloatSlider(value=1, min=0.001, max=1, step=0.03, description='sigma:')

# Use interact to create an interactive plot
interact(plot_dist_and_fit, sigma=sigma_slider, L_median=LM_slider)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Function to update the plot
def update_plot():
    amplitude = amplitude_var.get()
    frequency = frequency_var.get()
    phase = phase_var.get()
    func_type = func_type_var.get()
    
    x = np.linspace(0, 10, 1000)
    if func_type == 'sine':
        y = amplitude * np.sin(frequency * x + phase)
    else:
        y = amplitude * np.cos(frequency * x + phase)
    
    ax.clear()
    ax.plot(x, y, label=f'{func_type.capitalize()} Wave')
    ax.set_title(f'{func_type.capitalize()} Wave')
    ax.set_xlabel('x')
    ax.set_ylabel(f'{func_type}(x)')
    ax.legend()
    ax.grid(True)
    canvas.draw()

# Create the main window
root = tk.Tk()
root.title("Interactive Plot")

# Create matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create sliders and dropdown for parameters
controls_frame = ttk.Frame(root)
controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

amplitude_var = tk.DoubleVar(value=1.0)
frequency_var = tk.DoubleVar(value=1.0)
phase_var = tk.DoubleVar(value=0.0)
func_type_var = tk.StringVar(value='sine')

ttk.Label(controls_frame, text="Amplitude:").grid(row=0, column=0)
amplitude_slider = ttk.Scale(controls_frame, from_=0.1, to_=5.0, variable=amplitude_var, orient=tk.HORIZONTAL, command=lambda x: update_plot())
amplitude_slider.grid(row=0, column=1, sticky='ew')

ttk.Label(controls_frame, text="Frequency:").grid(row=1, column=0)
frequency_slider = ttk.Scale(controls_frame, from_=0.1, to_=5.0, variable=frequency_var, orient=tk.HORIZONTAL, command=lambda x: update_plot())
frequency_slider.grid(row=1, column=1, sticky='ew')

ttk.Label(controls_frame, text="Phase:").grid(row=2, column=0)
phase_slider = ttk.Scale(controls_frame, from_=0.0, to_=2*np.pi, variable=phase_var, orient=tk.HORIZONTAL, command=lambda x: update_plot())
phase_slider.grid(row=2, column=1, sticky='ew')

ttk.Label(controls_frame, text="Function:").grid(row=3, column=0)
func_type_dropdown = ttk.Combobox(controls_frame, textvariable=func_type_var, values=['sine', 'cosine'], state='readonly')
func_type_dropdown.grid(row=3, column=1, sticky='ew')
func_type_dropdown.bind("<<ComboboxSelected>>", lambda event: update_plot())

# Set the initial plot
update_plot()

# Start the Tkinter main loop
root.mainloop()

"""
# %%
