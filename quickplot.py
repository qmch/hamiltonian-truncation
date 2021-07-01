# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:34:23 2020

@author: aura1
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def powerlaw_offset(x, y0, a, n):
    return y0 + a*np.power(x,-n)

def main():
    lmax_vals = np.arange(1.5,8.5)
    mass_gap = [1.040524324172413,
                1.0178373805628915,
                1.0099935792170385,
                1.0063845297691891,
                1.0064220663290975,
                1.0032526281598741,
                1.00326253048339]
    # lmax_vals = np.arange(1,9)
    # mass_gap = [1.907381620393867,
    #             1.715060177638449,
    #             1.6803744291157905,
    #             1.6682371927439341,
    #             1.6624323199995565,
    #             1.6591860781893892,
    #             1.6571824919878146,
    #             1.6558575904310047]
    
    """
        Note: runtimes are <1 second up to lmax=3, and then 8.9
        seconds for lmax=4 and 39 seconds for lmax=5. Runtime
        increases to a little over 2 minutes for lmax=6
        (~134 seconds), and to about 8 minutes (467 seconds) for
        lmax=7. lmax=8 takes about half an hour to run.
        
        In the antiperiodic case, lmax=13/2 taks about 3 minutes (190 s) to run,
        and lmax=15/2 takes 8 minutes (475 s).
    """
    
    plt.xlabel("lmax")
    plt.ylabel("shifted mass gap (m/g-1)")
    plt.title("g=1,L=2pi, antiperiodic case")
    plt.xticks(np.arange(0,lmax_vals[-1]+1,0.5))
    
    #plt.grid(True)
    
    popt, pcov = curve_fit(powerlaw_offset, lmax_vals[1:], mass_gap[1:])
    print(f"Non-linearized curve fit: {popt}")
    
    plt.scatter(lmax_vals,np.array(mass_gap)-1)
    
    fit_xvals = np.linspace(lmax_vals[0],lmax_vals[-1])
    plt.loglog(fit_xvals,powerlaw_offset(fit_xvals,popt[0],popt[1],popt[2])-1,
             label=f"Non-linearized fit: y={popt[0]:.2f}+{popt[1]:.2f}/x^{popt[2]:.2f}")
    
    [y0, a, n] = gradient_curvefit(lmax_vals[1:], mass_gap[1:])
    print(f"Gradient curve fit: {[y0,a,n]}")
    plt.loglog(fit_xvals,powerlaw_offset(fit_xvals, y0, a, n)-1,
             label=f"Linearized fit: y={y0:.2f}+{a:.2f}/x^{n:.2f}")
    
    plt.legend()
    
    #plt.savefig('antiperiodic_mass_gap_with_fits_loglog_(Ian).pdf')
    
    plt.show()

def gradient_curvefit(xdata,ydata):
    """
    Fits a power law by first taking the numerical derivative
    to get rid of a constant offset, then linearizing the data
    to extract the power law behavior. Given the exponent, run
    the fit on the original data to determine the limiting behavior.

    Parameters
    ----------
    xdata : array of float
    ydata : array of float

    Returns
    -------
    powerlaw_params : array of float
        An array of three numbers:
            y0, the overall vertical offset
            a, the prefactor of the power law dependence
            n, the exponent in a/x**n

    """
    # the data has the form a+bx^(-n)
    # so the gradient has the form -bn*x^(-n-1)
    grad_ydata = np.gradient(ydata,xdata,edge_order=2)
    
    # taking the log to linearize, we have log(-bn) + (-n-1)*x
    logx = np.log10(xdata[:-1])
    # note that since n is positive, we have to take the log of
    # +bn*x^(n-1) instead
    logy = np.log10(-grad_ydata[:-1])
    
    linear = lambda x, m, y0: y0 + m*x
    popt, pcov = curve_fit(linear, logx, logy)
    
    n = -(popt[0] + 1)
    a = np.power(10,popt[1]) / n
    
    #print(f"n={n},a={a}")
    
    powerlaw_offset = lambda x, offset: offset + a*x**(-n)
    
    popt, pcov = curve_fit(powerlaw_offset, xdata, ydata)
    y0 = popt[0]
    #print(f"y0={popt[0]}")
    
    return [y0, a, n]

if __name__ == "__main__":
    main()
