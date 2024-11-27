
"""Exponential Fittin for the given data:

del_n (t) = \sum_{l=0}^N \exp^{-6 D_l t}* k_l

the experiment is repeated M times: 
t_1, t_2, ..., t_M
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
T = 320

D: float = 4.019
alpha: float = 0.366


def create_t(M: int = 1000) -> np.ndarray:
    return np.linspace(0, 3.5, M)


def create_del_n(M: int) -> np.ndarray:
    """D_0 is around 1'000 , D_N = 1 and the rest decreases exponentially"""
    t = create_t(M)
    del_n = np.exp(-(6*D*t)**alpha)
    return del_n

def create_l(N: int) -> np.ndarray:
    return np.linspace(5,1000,N)


def create_single_D(l: int) -> float:
    l = l*10**(-9)
    AR = 15
    eps = 1e-10
    eta = np.exp(-2.3568+617.2/(T-179.414))*10**-2  # 2*10**(-1)
    kT = 4.11*10**(-21)/300 * T
    S = np.log(AR+np.sqrt(AR**2-1)) / np.sqrt(AR**2-1)
    # added epsilon otherwise division by zero
    A = (3*kT)/(np.pi*eta*(l+eps)**3)
    G = (0.5*AR**2*(((2*AR**2-1) / AR)*S-1)) / (AR**2-1)
    D = A*G
    return D


def create_D(N: int) -> np.ndarray:
    return np.array([create_single_D(l) for l in create_l(N)])


def create_exp_matrix(N: int, M: int) -> np.ndarray:
    vec_t = create_t(M)
    vec_D = create_D(N)
    return np.array([[np.exp(-6*vec_D[i]*vec_t[j])
                      for i in range(N)] for j in range(M)])


def main():
    # Data
    M = 70000
    for N in [100]:
        
    
        vec_t = create_t(M) #vec_t = np.array([i/M for i in range(M)])
        vec_del_n = create_del_n(M)
        vec_D = create_D(N)

        # Create matrix
        mat_exp = create_exp_matrix(N, M)
        vec_k = np.linalg.lstsq(create_exp_matrix(N, M),vec_del_n,)
        vec_k = vec_k[0]
        # each entry is np.exp(-6*D_i*t_j)
        # mat_exp = np.matrix([[np.exp(-6*vec_D[i]*vec_t[j])
        #                    for j in range(M)] for i in range(N)])

        # vector of k is vector of length N
        # vec_k = np.linalg.pinv(mat_exp) * vec_del_n
        # print("vec_k: ", vec_t[:5])
        # print("vec_D: ", vec_D[:5])
        # print("vec_del_n: ", vec_del_n[0:5])
        print(mat_exp)
        print(vec_k)
        plt.plot( create_l(N), np.abs(vec_k)/sum(np.abs(vec_k[5:])), marker = '.',label = f'{N}')
    vec_real_dist = np.abs(vec_k)/sum(np.abs(vec_k[5:]))
    
    
    AR = 15
    V = 0.25*np.pi*(create_l(N)**3/AR**2)

    vec_real_dist = vec_real_dist/V
    vec_real_dist = vec_real_dist/sum(vec_real_dist)
    plt.plot( create_l(N), vec_real_dist, marker = '.',label = f'{N}')
    plt.legend()
    plt.grid()
    #plt.ylim(0, 0.5)
    plt.show()


    # RECONSTRUCT THE EXP
  

    








if __name__ == "__main__":
    main()




# %%
