import numpy as np
import matplotlib.pyplot as plt

def rot_diff_par(l,AR) :
    
    eta = 1.6*10**(-1)	  #1.61*10^(-2);
    kT = 4.11*10**(-21)
    S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
    A = (3*kT)/(np.pi*eta*l**3)
    G = (0.5*AR**2*( ((2*AR**2-1) / AR)*S-1) ) / (AR**2-1)
    D = A*G

    return D

def D_qilin(l,AR):
    n = 16
    A = (3*1.38065*298)/(2*np.pi*n*l**3)*(-1+2*np.log(2*AR))*10**6
    return A

def D_marci(l,AR) :
    
    eta = 1.6*10**(-1)	  #1.61*10^(-2);
    kT = 4.11*10**(-21)
    D = (3*kT)/(2*np.pi*eta*l**3)*(-1+2*np.log(2*AR))
    return D

lengths = np.linspace(20,200,1500)

ar = 25
D_q = D_qilin(lengths,ar)
D_other_paper = rot_diff_par(lengths*10**(-9),ar)
D_m = D_marci(lengths*10**(-9),ar)

plt.plot(lengths,D_q, label = 'qilin')
plt.plot(lengths,D_other_paper, label = 'other paper')
plt.plot(lengths,D_m, label = 'marcello')
plt.legend()
plt.grid('both')
plt.show()

