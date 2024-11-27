#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the log-normal PDF
def lognorm_pdf(x, sig, LM):
    return (1 / (x * sig * np.sqrt(2 * np.pi))) * np.exp(-np.log(x / LM)**2 / (2 * sig**2))

def lognorm_eff_correction(dist,lengths,ar):
    V = np.pi * ((0.5 * lengths / ar)**2) * lengths
    eff_dist = V * dist
    eff_dist = eff_dist / sum(eff_dist)
    return eff_dist
# Define the stretched exponential function
def st_exponential(x, b, alpha):
    return np.exp(-(b * x)**alpha)

# Function to fit the stretched exponential to data
def SE_from_TEBcurve(time, curve):
    if curve[0] != 1:
        curve = curve / curve[0]
    params, covariance = curve_fit(st_exponential, time, curve)
    b_fit, alpha_fit = params
    var_b, var_alpha = np.diag(covariance)
    err_b = np.sqrt(var_b)
    err_alpha = np.sqrt(var_alpha)

    # Generate the fitted curve using the parameters
    fit_curve = st_exponential(time, b_fit, alpha_fit)
    
    return fit_curve, b_fit / 6, alpha_fit


# Define the rotational diffusion parameter function
def rot_diff_par(l, AR, T=300):
    eta = np.exp(-2.3568 + 617.2 / (T - 179.414)) * 10**-2
    kT = 4.11 * 10**-21 / 300 * T
    S = np.log(AR + np.sqrt(AR**2 - 1)) / np.sqrt(AR**2 - 1)
    A = (3 * kT) / (np.pi * eta * l**3)
    G = (0.5 * AR**2 * (((2 * AR**2 - 1) / AR) * S - 1)) / (AR**2 - 1)
    D = A * G
    return D

def generate_TEB(t,eff_dist,lengths,ar):

    
    D_l = rot_diff_par(lengths * 10**(-9), ar)

    exps = [np.exp(-6 * D_l[i] * t) for i in range(len(D_l))]
    w_exps = [eff_dist[i] * exps[i] for i in range(len(eff_dist))]

    relax = [sum(w_exps[i][j] for i in range(len(D_l))) for j in range(len(t))]

    relax_corrected_EG = relax + 0.075 * np.exp(-(6 * 0.4 * t)**0.8)
    relax_corrected_EG = relax_corrected_EG / relax_corrected_EG[0]
    
    return relax_corrected_EG



# %%

#%% Make table D, alpha and L and sigma
LM_10_D =[]
LM_30_D =[]
LM_50_D =[]
LM_70_D =[]
LM_90_D =[]
LM_110_D =[]
LM_10_a =[]
LM_30_a =[]
LM_50_a =[]
LM_70_a =[]
LM_90_a =[]
LM_110_a =[]
LM_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 ]
sigma_list = [0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
#%%
with open('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_D.txt', 'w') as f:
    f.write('') 
with open('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_alpha.txt', 'w') as f:
    f.write('')
for LM in LM_list:
    
    for sigma in sigma_list:

        
        lengths = np.linspace(5, 1000, 1000)
        ar = 3.34885e-5*lengths**2+1.16861256e-1*lengths+7.594886
        t = np.linspace(0, 2, 2000)


        dist = lognorm_pdf(lengths,sigma,LM)
        eff_dist = lognorm_eff_correction(dist,lengths,ar)
        TEB = generate_TEB(t,eff_dist,lengths,ar)
        SE_fit,D_fit,alpha_fit =  SE_from_TEBcurve(t, TEB)

        if LM == 10: LM_10_D.append(D_fit)
        if LM == 10: LM_10_a.append(alpha_fit)

        if LM == 30: LM_30_D.append(D_fit)
        if LM == 30: LM_30_a.append(alpha_fit)
        
        if LM == 50: LM_50_D.append(D_fit)
        if LM == 50: LM_50_a.append(alpha_fit)
 
        if LM == 70: LM_70_D.append(D_fit)
        if LM == 70: LM_70_a.append(alpha_fit)

        if LM == 90: LM_90_D.append(D_fit)
        if LM == 90: LM_90_a.append(alpha_fit)

        if LM == 110: LM_110_D.append(D_fit)
        if LM == 110: LM_110_a.append(alpha_fit)

        with open('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_D.txt', 'a') as f:
            f.write(f'{D_fit}   ')
        with open('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_alpha.txt', 'a') as f:
            f.write(f'{alpha_fit}   ')

    with open('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_D.txt', 'a') as f:
            f.write('\n')
    with open('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_alpha.txt', 'a') as f:
            f.write('\n')
#%%
'''
fig, ax1 = plt.subplots()



ax1.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_30_D,linestyle = '-',marker = '$D$',label = 'Lm = 30nm')
ax1.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_50_D,linestyle = '-',marker = '$D$',label = 'Lm = 50nm')
ax1.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_70_D,linestyle = '-',marker = '$D$',label = 'Lm = 70nm')
ax1.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_90_D,linestyle = '-',marker = '$D$',label = 'Lm = 90nm')
ax1.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_110_D,linestyle = '-',marker = '$D$',label = 'Lm = 110nm')
ax1.set_ylabel('D')ax1.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_10_D,linestyle = '-',marker = '$D$',label = 'Lm = 10nm')
ax1.set_xlabel(r'$\sigma$')
ax1.set_yscale('log')
lines_1, labels_1 = ax1.get_legend_handles_labels()

ax1.legend(lines_1, labels_1 , loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)



ax2 = ax1.twinx()

ax2.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_10_a,linestyle = '-',marker = r'$\alpha$',label = 'Lm = 10nm')
ax2.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_30_a,linestyle = '-',marker = r'$\alpha$',label = 'Lm = 30nm')
ax2.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_50_a,linestyle = '-',marker = r'$\alpha$',label = 'Lm = 50nm')
ax2.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_70_a,linestyle = '-',marker = r'$\alpha$',label = 'Lm = 70nm')
ax2.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_90_a,linestyle = '-',marker = r'$\alpha$',label = 'Lm = 90nm')
ax2.plot([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], LM_110_a,linestyle = '-',marker = r'$\alpha$',label = 'Lm = 110nm')

ax2.set_ylabel(r'$\alpha$')
ax2.set_xlabel(r'$\sigma$')

ax1.axhline(y=3.195, color='r', linestyle='--')
ax2.axhline(y=0.61, color='b', linestyle='--')
plt.show()

'''
#%%

bitmap_D  = np.loadtxt('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_D.txt')
rows, cols = bitmap_D.shape
for i in range(rows):
    for j in range(cols):
        value = np.log(bitmap_D[i, j])
        formatted_value = f"{value:.2f}"  # Format to show two decimal places
        plt.text(j, i, formatted_value, ha='center', va='center', color='red', fontsize=5)

plt.imshow(np.log(bitmap_D), interpolation='nearest')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$L_{M}$')
plt.title('log D')
plt.colorbar()  # Add a colorbar to the plot
plt.xticks(ticks=np.arange(0,21,1),labels=sigma_list,fontsize = 6)
plt.yticks(ticks=np.arange(0,20,1),labels=LM_list)
plt.show()

bitmap_alpha  = np.loadtxt('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/Results/Biref/lm_sigma_to_alpha.txt')
rows, cols = bitmap_alpha.shape
for i in range(rows):
    for j in range(cols):
        value = (bitmap_alpha[i, j])
        formatted_value = f"{value:.2f}"  # Format to show two decimal places
        plt.text(j, i, formatted_value, ha='center', va='center', color='red', fontsize=5)

plt.imshow((bitmap_alpha), interpolation='nearest')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$L_{M}$')
plt.title(r'$\alpha$')
plt.colorbar()  # Add a colorbar to the plot
plt.xticks(ticks=np.arange(0,21,1),labels=sigma_list, fontsize=6  )
plt.yticks(ticks=np.arange(0,20,1),labels=LM_list)
plt.show()
#%%
def find_min_line(mat,val):
    
    rows, cols = mat.shape
    positions = np.zeros(cols)
    for j in range(cols):
        closer_values  = [b for b in mat[:,j]]
        for i in range(1,rows):
            if np.abs(val - closer_values[i]) < np.abs(val - closer_values[i-1]):
                positions[j]  = i
        print(f'col {j} closer val {closer_values[int(positions[j])]} at pos {int(positions[j])}')
    invertet_pos = [np.abs(p-cols+1) for p in positions]
    plt.plot(sigma_list,invertet_pos)
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$L_{M}$')

    plt.grid('both')
    plt.show()

    return invertet_pos
#%%

minline_D = find_min_line(np.log(bitmap_D),`np.log(1.311)`)
minline_alpha  = find_min_line(bitmap_alpha,0.462)
4#%%
plt.plot(sigma_list,minline_D,label='D')
plt.plot(sigma_list,minline_alpha,label='alpha')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$L_{M}$')

plt.grid('both')
plt.legend()
plt.show()
   #%%

# Function to generate data and update the plot
def generate_data_and_update_plot(sigma, L_median):
    lengths = np.linspace(5, 1000, 1000)
    ar = 15

    dist = lognorm_pdf(lengths, sigma, L_median)
    V = np.pi * ((0.5 * lengths / ar)**2) * lengths
    eff_dist = V * dist
    eff_dist = eff_dist / sum(eff_dist)

    t = np.linspace(0, 2, 2000)
    D_l = rot_diff_par(lengths * 10**(-9), ar)

    exps = [np.exp(-6 * D_l[i] * t) for i in range(len(D_l))]
    w_exps = [eff_dist[i] * exps[i] for i in range(len(eff_dist))]

    relax = [sum(w_exps[i][j] for i in range(len(D_l))) for j in range(len(t))]

    relax_corrected_EG = relax + 0.1 * np.exp(-(6 * 0.4 * t)**0.8)
    relax_corrected_EG = relax_corrected_EG / relax_corrected_EG[0]

    SE, D_eff, alpha = SE_from_TEBcurve(t, relax_corrected_EG)

    ax1.clear()
    ax2.clear()

    ax1.plot(lengths[:500], dist[:500], label=f'sigma={sigma} L_median={L_median}')
    ax2.plot(t, relax_corrected_EG, label=f'alpha={alpha:.3f}, D_eff={D_eff:.3e}')
    ax2.plot(t, SE, ':')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Normalized Biref')
    ax2.legend()
    ax2.grid('both')

    ax1.set_xlabel('Length')
    ax1.set_ylabel('Size Distribution')
    ax1.legend()
    ax1.grid('both')

    fig.tight_layout()
    canvas.draw()

# Update plot function
def update_plot(event=None):
    sigma = sigma_var.get()
    L_median = L_median_var.get()
    generate_data_and_update_plot(sigma, L_median)

# Create the main window
root = tk.Tk()
root.title("Interactive Plot with Tkinter")

# Create matplotlib figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create sliders for sigma and L_median
controls_frame = ttk.Frame(root)
controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

sigma_var = tk.DoubleVar(value=0.3)
L_median_var = tk.DoubleVar(value=120)

ttk.Label(controls_frame, text="Sigma:").grid(row=0, column=0)
sigma_slider = ttk.Scale(controls_frame, from_=0.001, to_=1.0, variable=sigma_var, orient=tk.HORIZONTAL, command=update_plot)
sigma_slider.grid(row=0, column=1, sticky='ew')

ttk.Label(controls_frame, text="L_median:").grid(row=1, column=0)
L_median_slider = ttk.Scale(controls_frame, from_=5.0, to_=300.0, variable=L_median_var, orient=tk.HORIZONTAL, command=update_plot)
L_median_slider.grid(row=1, column=1, sticky='ew')

# Set the initial plot
generate_data_and_update_plot(sigma_var.get(), L_median_var.get())

# Start the Tkinter main loop
root.mainloop()

# %%
