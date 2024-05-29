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


# %%
def find_mean(distribution) :
    distribution = distribution / np.sum(distribution)
    
    mean_value = np.sum(np.arange(0,len(distribution),1)*distribution)
    
    return mean_value

def find_median(distribution):
    
    # Calculate the cumulative sum of probabilities
    cumulative_probs = np.cumsum(distribution)
    
    # Find the index where cumulative sum is closest to half of total sum
    median_index = np.abs(cumulative_probs - np.sum(distribution) / 2).argmin()
    
    # Return the value at median index
    return median_index

def rot_diff_par(l,AR) :
    
    T = 300
    eta = np.exp(-2.3568+617.2/(T-179.414))*10**-2   #2*10**(-1)	  
    kT = 4.11*10**(-21)/300 * T
    S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
    A = (3*kT)/(np.pi*eta*l**3)
    G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
    D = A*G
    return D

file_path = 'C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/aspectRATIOr.txt'  # replace with the actual file path

dat = np.loadtxt(file_path,  delimiter='\t', usecols=(2), unpack=True)

dat = np.abs(dat)

l = []
w = []
for i in range(0,len(dat),2):
    l.append(dat[i])
for i in range(1,len(dat),2):
    w.append(dat[i])

x1 = np.arange(0,len(l)) 
x2 = np.arange(0,2*len(w),2) 

AR = []
for i in range(len(l)):
    AR.append(l[i]/w[i])

l = np.array(l)
AR = np.array(AR)
#plt.plot(AR, label = "AR")
#plt.show()


l = 0.000001*l  #conversion to m from um

l = 10**9*l # l in nm


def lin(x,m,q):
    return m*x+q

def quad(x , a,b,c) :
    return a*x**2+b*x+c
abc, abcpcov = curve_fit(quad,l,AR)
qm, qmpcov = curve_fit(lin,l,AR)

def ar(l):
    aspect_ratio = abc[0]*l**2+abc[1]*l+abc[2]
    #aspect_ratio = qm[0]*l+qm[1]
    return aspect_ratio

def ar_max(l):
    aspect_ratio = (abc[0]+np.sqrt(abcpcov[0,0]))*l**2+(abc[1]+np.sqrt(abcpcov[1,1]))*l+abc[2]+np.sqrt(abcpcov[2,2])

    #aspect_ratio = (qm[0]+np.sqrt(qmpcov[0,0])) * l + qm[1]+np.sqrt(qmpcov[1,1])
    return aspect_ratio 

def ar_min(l):
    aspect_ratio = (abc[0]-np.sqrt(abcpcov[0,0]))*l**2+(abc[1]-np.sqrt(abcpcov[1,1]))*l+abc[2]-np.sqrt(abcpcov[2,2])
    #aspect_ratio = (qm[0]-np.sqrt(qmpcov[0,0])) * l -np.sqrt(qmpcov[1,1])
    for a in range(len(aspect_ratio)):
        if aspect_ratio[a] < 2 : aspect_ratio[a] = 2
    return aspect_ratio

def ar_chosen(l):
    aspect_ratio = (abc[0]-np.sqrt(abcpcov[0,0]))*l**2+(abc[1]-np.sqrt(abcpcov[1,1]))*l+abc[2]
    return aspect_ratio
plot_lAR = True
if plot_lAR == True :
    plt.scatter(l,AR)
    plt.plot( np.arange(min(l),700), ar( np.arange( min(l),700)  ) ,color='r',label = f'{qm[0]}  {qm[1]}' )
    plt.plot(np.arange(min(l),700), ar_max(np.arange(min(l),700)) ,color='b')
    plt.plot(np.arange(min(l),max(l)), ar_min(np.arange(min(l),max(l))) ,color='g')
    plt.plot(np.arange(min(l),max(l)), ar_chosen(np.arange(min(l),max(l))))
    plt.legend()
    plt.xlabel('legths [nm]')
    plt.ylabel('AR')
    plt.show()

print(f'AR = {qm[0]}l {qm[1]}')

# %% testtttttttttttttttttttttttttttttttttttttttt
# Create histogram
hist, bins = np.histogram(l,bins=30, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2


#dist = fit_dist(bin_centers, hist, 5)
cum_dist = np.cumsum(hist)


# Fit a polynomial of the specified degree to the data
coefficients = np.polyfit(bin_centers, cum_dist,5)

# Create a polynomial object        
polynomial = np.poly1d(coefficients)

# Generate x values for plotting the fit
x_fit = np.linspace(bins.min(), bins.max(), 1000)
fit_cum_dist = polynomial(x_fit)

# Plot the original data
plt.scatter(bin_centers, cum_dist, label='Data')

# Plot the polynomial fit
plt.plot(x_fit, fit_cum_dist, label=f'Polynomial Fit (degree {5})', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Fit to Data')
plt.show()

dist = np.diff(fit_cum_dist)

plt.hist(l,bins=30, density = True)

plt.plot(x_fit[0:999], dist*max(hist)/max(dist))
plt.show()

# %%TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

# Define a skewed normal distribution function
def skew_normal_pdf(x, loc, scale, skew):
    return skewnorm.pdf(x, a=skew, loc=loc, scale=scale)*skew*x

# Fit the data to the skewed normal distribution
popt, pcov = curve_fit(skew_normal_pdf, bin_centers, hist, p0=[np.mean(l), np.std(l), 0])


def correction(lengths,asp_rat, loc, scale, skew):
    Surface =  2* np.pi*lengths/asp_rat*lengths
    V = np.pi*( (0.5*lengths/asp_rat)**2)*lengths
    new_dist = V*skewnorm.pdf(lengths, a=skew, loc=loc, scale=scale)*skew*lengths
    new_dist  = new_dist/sum(new_dist)
    return new_dist
#%%
# Plot the fitted curve
x_range = np.linspace(min(l), max(l), 100)
dist = skew_normal_pdf(x_range, *popt)
eff_dist = correction(x_range,20, *popt)
#eff_dist =  np.pi*((0.5*ar(x_range)/x_range)**2)*x_range*skew_normal_pdf(x_range, *popt)

plt.hist(l, bins=30, density=True, alpha=0.5, label='lengths distribution')
plt.plot(x_range, dist/sum(dist), color = 'b', label='fitted distribution')
plt.plot(x_range, eff_dist/(sum(eff_dist)), color = 'r', label='fitted distribution*Vol')
plt.axvline(  x = np.argmax(eff_dist) * (max(l)-min(l))/100 +min(l), color = 'r', linestyle='--', alpha = 0.7 )
plt.axvline(  x = np.argmax( skew_normal_pdf(x_range, *popt) ) * (max(l)-min(l))/100 +min(l), color = 'b', linestyle='--', alpha = 0.7 )
plt.xlabel('length [nm]')
plt.legend()
plt.show()

# %%
ldist = 1500
dist = skew_normal_pdf(np.arange(20,ldist,1), *popt)
eff_dist = correction(np.arange(20,ldist,1),20, *popt)
print('>>> MODE :')
effective_l_mode = np.argmax( eff_dist ) + 20
effective_l_mode = effective_l_mode*10**(-9)

mode_l = np.argmax( dist ) +20
mode_l = mode_l*10**(-9)
effective_D_mode = rot_diff_par(effective_l_mode,ar(mode_l*10**9)) #( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*effective_l_mode)**3)
calculated_D_mode = rot_diff_par(mode_l,ar(mode_l*10**9)) #( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*mode_l)**3)

print('effective l[nm] = ',effective_l_mode*10**9)
print('calculated l[nm] = ',mode_l*10**9)

print('effective Diff coeff = ',effective_D_mode)
print('calculated Ddiif coeff = ',calculated_D_mode)

print('.......................................................')


print('>>> MEDIAN :')
effective_l_median = find_median(eff_dist)+20 #np.abs(eff_dist - find_median(eff_dist)).argmin()
effective_l_median = effective_l_median*10**(-9) #trsform to m for D calculation

median_l = find_median(dist)+20
median_l = median_l*10**(-9)

effective_D_median = rot_diff_par(effective_l_median,ar(median_l*10**9))#( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*effective_l_median)**3)
calculated_D_median = rot_diff_par(median_l,ar(median_l*10**9))#( 3*kT*(-1+2*np.log(np.mean(AR))) )/(16*np.pi*eta*(0.5*median_l)**3)

print('effective l[nm] = ',effective_l_median*10**9) #trasform to nm for clarity
print('calculated l[nm] = ',median_l*10**9)

print('effective Diff coeff = ',effective_D_median)
print('calculated Diif coeff = ',calculated_D_median)

print('.......................................................')

print('>>> MEAN/AVERAGE :')
effective_l_average = find_mean(eff_dist)+20
effective_l_average = effective_l_average*10**(-9)

average_l = find_mean(dist)+20
average_l = average_l*10**(-9)

effective_D_average = rot_diff_par(effective_l_average,ar(average_l*10**9))#( 3*kT*(-1+2*np.log(np.mean(2*AR))) )/(16*np.pi*eta*(0.5*effective_l_average)**3)
calculated_D_average = rot_diff_par(average_l,ar(average_l*10**9)) #( 3*kT*(-1+2*np.log(np.mean(2*AR))) )/(16*np.pi*eta*(0.5*average_l)**3)



print('effective l[nm] = ',effective_l_average*10**9)
print('calculated l[nm] = ',average_l*10**9)

print('effective Diff coeff = ',effective_D_average)
print('calculated Ddiif coeff = ',calculated_D_average)

print('.......................................................')

#%%
#allah uh akbar
plt.plot(np.arange(20,ldist,1),eff_dist/sum(eff_dist), linestyle = '-.' ) 
plt.axvline(x=effective_l_mode*10**9, linestyle = '-', color = 'b', label = 'mode')
plt.axvline(x=effective_l_median*10**9, linestyle = '-.' ,color = 'b', label = 'median')
plt.axvline(x=effective_l_average*10**9, linestyle = ':' , color = 'b', label = 'mean')
plt.legend()

plt.plot(np.arange(20,ldist,1),dist ) 
plt.axvline(x=mode_l*10**9, linestyle = '-',color = 'r')
plt.axvline(x=median_l*10**9, linestyle = '-.',color = 'r')
plt.axvline(x=average_l*10**9, linestyle = ':',color = 'r')
plt.legend()
plt.xlim(0,800)
plt.xlabel('length [nm]')
plt.show()


# %%
#plot D dist
lengths = np.arange(100,800,1)
D = rot_diff_par(lengths*10**(-9),ar(lengths))
D_max = rot_diff_par(lengths*10**(-9),ar_max(lengths))
D_min = rot_diff_par(lengths*10**(-9),ar_min(lengths))
#D_agg = rot_diff_par(lengths*10**(-9),2)
#D_ar30 = rot_diff_par(lengths*10**(-9),30)
plt.plot(lengths,D)
#plt.plot(lengths,D_max)
#plt.plot(lengths,D_min)
#plt.plot(lengths,D_agg, label = 'AR = 10')
#plt.plot(lengths,D_ar30, label = 'AR 30')
plt.axvline(x=mode_l*10**9, color = 'r', label = 'mode')
plt.axvline(x=median_l*10**9, color = 'b', label = 'median')
plt.axvline(x=average_l*10**9, color = 'g', label = 'mean')
plt.axvline(x=effective_l_mode*10**9, linestyle = '--', color = 'r', label = 'mode')
plt.axvline(x=effective_l_median*10**9, linestyle = '--', color = 'b', label = 'median')
plt.axvline(x=effective_l_average*10**9, linestyle = '--', color = 'g', label = 'mean')
plt.axhline(4, color = 'gray' , alpha = 0.5)
plt.legend()
plt.ylim(0,50)
plt.xlabel('length [nm]')
plt.ylabel(r'Rotational diffusion parameter [$s^{-1}$]')
plt.xlim(100,500)
plt.grid('minor')
plt.show()

print('*******************************************************')
# %% CAN WE COMPUTE THE CURVE FROM THE RELAXATION?
    
t = np.linspace(0,2,1000)
lengths =  np.linspace(5,500,10)
ar = 10

def correction(lengths,asp_rat, loc, scale, skew):
    #Surface =  2* np.pi*lengths/asp_rat*lengths
    V = np.pi*( (0.5*lengths/asp_rat)**2)*lengths
    new_dist = V*skewnorm.pdf(lengths, a=skew, loc=loc, scale=scale)*skew*lengths
    return new_dist/sum(new_dist)

for T in [300]:
    def rot_diff_par(l,AR) :
    
        eta = np.exp(-2.3568+617.2/(T-179.414))*10**-2   #2*10**(-1)	  
        kT = 4.11*10**(-21)/300 * T
        S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
        A = (3*kT)/(np.pi*eta*l**3)
        G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
        D = A*G
        return D

    eff_dist =  correction(lengths,ar, *popt)
    num_dist = skew_normal_pdf(lengths, *popt)
    num_dist = num_dist/sum(num_dist)
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

    relax = relax

    # non corrected
    w_exps_nc = []
    for i in range(len(num_dist)) :
     w_exps_nc.append(num_dist[i]*exps[i])


    relax_nc = []
    somma_exps_nc = 0
    for j in range(len(t)):
        for i in range(len(D_l)):
            somma_exps_nc = somma_exps_nc  + w_exps_nc[i][j]
        relax_nc.append(somma_exps_nc)
        somma_exps_nc = 0

    relax_nc = relax_nc/relax_nc[0]

    # just taking the measurements i did.
    DD = 4

    plt.plot(t,relax, marker = '.', label = f'eff dist')
    #plt.plot(t,relax_min, linestyle = '-.', label = 'MIN effective distribution')
    #plt.plot(t,relax_max, linestyle = '-.', label = 'MAX effective distribution')
    plt.plot(t,relax_nc,marker = '.',linestyle = 'none', label = f'only number dist T={T}K')
    #plt.plot(t,np.exp(-6*DD*t), label = 'single exp' )

plt.grid('both')

# decide dataset to plot with 


plt.plot(t, np.exp(-(6*4*t)**0.3),alpha = 0.7, color = 'r',  label = 'from EC')
plt.xlim(-0.05,1.75)
plt.legend()
plt.xlabel('time[s]')
plt.grid('both')
plt.show()


# %%
'''


t = np.linspace(0,3.5, 1000)
data = np.exp(-(6*4.019*t)**0.366)


def SE(x,Dr,a):
    return np.exp(-(6*Dr*x)**a)

Dra, DraCOV = curve_fit(SE,t,data)

# given an aspect ratio at choice find to what effective lenngth this value of D corresponds

aspect_ratio = 20 #could be an array long like l

def rot_diff_par(l,AR) :
    
    eta = np.exp(-2.3568+617.2/(T-179.414))*10**-2   #2*10**(-1)	  
    kT = 4.11*10**(-21)/300 * T
    S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
    A = (3*kT)/(np.pi*eta*l**3)
    G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
    D = A*G
    return D

Lm_arg = np.abs(rot_diff_par(lengths*10**(-9),aspect_ratio)-Dra[0]).argmin()
Lm = lengths[Lm_arg]

lbins = np.arange(Lm-300,Lm+200,50) #manually chose the binning

#create multiexp function ME
Dl = rot_diff_par(lbins*10**(-9),aspect_ratio)
def ME(time,k):
    k = k*1
    sum = np.exp(-6*Dl[0]*time)
    for i in range(0,len(Dl)):
        sum = sum + np.exp(-6*Dl[i]*time)
    return sum

K, meCOV = curve_fit(ME,t,data)
plt.plot(lbins,K)
plt.show()

 # %%
'''
# %%
