######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import schwinger
import sys
import scipy
import time
from scipy.constants import pi

def main(argv):
    
    #if there are too few arguments, print the right syntax and exit
    if len(argv) < 3:
        print("python genMatrix.py <R> <Emax>")
        sys.exit(-1)
    
    print("Beginning execution.")
    startTime = time.time()
    #circle circumference (p.5)
    R = float(argv[1])
    #maximum energy (compare Eq. 2.26)
    Emax = float(argv[2])
    #mass
    m = 0.
    
    a = schwinger.Schwinger()
    
    a.buildFullBasis(2*pi*R, m, Emax)

    print(f"Basis size: {a.fullBasis.size}")
    # print(f"Basis elements: {a.fullBasis}")

    #set the file name for saving the generated matrix
    fstr = "Emax="+str(a.fullBasis.Emax)+"_L="+str(a.L)

    a.buildMatrix()
    
    a.buildBasis(a.fullBasis.Emax)
    
    computeVacuumEnergy(a,g=1)
    
    print(f"Spectrum: {a.spectrum()}")
    #print(a.h0)
    #print(a.fullBasis)
    print("Runtime:",time.time()-startTime)
    #a.saveMatrix(fstr)

def computeVacuumEnergy(schwinger, g):
    """
    

    Parameters
    ----------
    phi4 : Phi1234
        A Phi1234 object initialized with a suitable basis and the matrices
        for the various terms in the Hamiltonian.
    g : float
        Value of the coupling g.

    Returns
    -------
    A float representing the vacuum energy for this set of parameters.

    """
    sigma = -30.
    neigs = 5

    schwinger.setcouplings(g)

    print(f"Computing raw eigenvalues for g4 = {g}")

    schwinger.computeHamiltonian(ren=False)

    schwinger.computeEigval(sigma=sigma, n=neigs, ren=False)
        
    vacuumEnergy = schwinger.vacuumE(ren="raw")
    print("Raw vacuum energy: ", vacuumEnergy)
    
    return vacuumEnergy

if __name__ == "__main__":
    main(sys.argv)
