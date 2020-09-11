######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import sys
import math
from math import log, pi
from scipy import integrate
import numpy
import scipy

"""
These functions correspond to kappa_0, kappa_2, and kappa_4 as given in Eq. 3.34
in arXiv:1412.3460v6. Given the original couplings g2 and g4 as well as a
reference energy (taken to be the vacuum energy in the raw truncation), we
can compute the effect on the couplings of integrating out the high-energy
states rather than just truncating the basis, and then numerically diagonalize
with respect to the renormalized couplings.
"""
def ft0(g2, g4, E, cutoff=0.):
    if E<cutoff:
        return 0.
    else:
        return (g2**2/pi + g4**2*(-3/(2*pi) + 18/pi**3 * log(E)**2))/(E**2)


def ft2(g2, g4, E, cutoff=0.):
    if E<cutoff:
        return 0.
    else:
        return (g2*g4*12/pi + g4**2*72/pi**2 * log(E))/(E**2)


def ft4(g2, g4, E, cutoff=0.):
    if E<cutoff:
        return 0.
    else:
        return (g4**2 * 36/pi) / (E**2)


def renlocal(g2, g4, Emax, Er):
    g0r = - integrate.quad(lambda E: ft0(g2,g4,E)/(E-Er),Emax,numpy.inf)[0]
    g2r = g2 - integrate.quad(lambda E: ft2(g2,g4,E)/(E-Er),Emax,numpy.inf)[0]
    g4r = g4 - integrate.quad(lambda E: ft4(g2,g4,E)/(E-Er),Emax,numpy.inf)[0]

    return [g0r,g2r,g4r]

def rensubl(g2, g4, Ebar, Emax, Er, cutoff):
    ktab = scipy.linspace(0.,Emax,30,endpoint=True)
    rentab = [[0.,0.,0.] for k in ktab]
    
    for i,k in enumerate(ktab):
        g0subl = - integrate.quad(lambda E: ft0(g2,g4,E-k,cutoff)/(E-Ebar) - ft0(g2,g4,E,cutoff)/(E-Er),Emax,numpy.inf)[0]
        g2subl = - integrate.quad(lambda E: ft2(g2,g4,E-k,cutoff)/(E-Ebar) - ft2(g2,g4,E,cutoff)/(E-Er),Emax,numpy.inf)[0]
        g4subl = - integrate.quad(lambda E: ft4(g2,g4,E-k,cutoff)/(E-Ebar) - ft4(g2,g4,E,cutoff)/(E-Er),Emax,numpy.inf)[0]
        
        rentab[i] = [g0subl, g2subl, g4subl]
        
    return ktab,numpy.array(rentab)

