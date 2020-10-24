# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:34:23 2020

@author: aura1
"""


import numpy as np
import matplotlib.pyplot as plt

def main():
    # lmax_vals = np.arange(1.5,7.5)
    # mass_gap = [1.04,1.02,1.009,1.006,1.006,1.003]
    #lmax_vals = np.arange(1,6)
    #mass_gap = [1.907,1.715,1.680,1.668,1.662]
    gvals = 
    
    plt.xlabel("lmax")
    plt.ylabel("mass gap (m/g)")
    plt.title("g=1,L=2pi, antiperiodic case")
    plt.xticks(np.arange(0,8,0.5))
    
    plt.plot(lmax_vals,mass_gap)
    plt.savefig('antiperiodic_mass_gap_(Ian).pdf')
    
    plt.show()
    
if __name__ == "__main__":
    main()
