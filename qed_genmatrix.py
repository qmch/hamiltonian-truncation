######################################################
# 
# Adaptation of phi4 methods for Hamiltonian truncation to QED/Schwinger model
#
######################################################

import schwinger
import sys
import scipy
import time
import sys
import numpy as np
from scipy.constants import pi

def main(argv):
    
    #if there are too few arguments, print the right syntax and exit
    if len(argv) < 3:
        print("python qed_genMatrix.py <R> <Emax>")
        sys.exit(-1)
    
    print("Beginning execution.")
    startTime = time.time()
    R = float(argv[1])
    Emax = float(argv[2])
    #mass
    m = 0.
    
    myBCs = "antiperiodic"
    
    a = schwinger.Schwinger()
    
    a.buildFullBasis(2*pi*R, m, Emax, bcs=myBCs)

    print(f"Basis size: {a.fullBasis.size}")
    # print(f"Basis elements: {a.fullBasis}")

    #set the file name for saving the generated matrix
    fstr = f"Emax={a.fullBasis.Emax}_R={R}_bcs={myBCs}"
    print(f"filename: {fstr}")

    a.buildMatrix()
    
    print("Runtime:",time.time()-startTime)
    
    a.saveMatrix(fstr)
    
    #temporary: also load in the same script
    #move this later
    """
    print("Beginning eigenvalue calculation.")
    startTime = time.time()
    
    #fname = argv[1]
    g = 1.#float(argv[2])
    Emax = 4.#float(argv[3])

    # Hardcoded parameters
    m=1.
    sigma = -30.
    neigs = 3

    b = schwinger.Schwinger()
    b.loadMatrix(fstr+".npz")

    b.buildBasis(Emax=Emax)
    
    print(f"full basis size = {b.fullBasis.size}")
    print(f"basis size = {b.basis.size}")
    
    b.setcouplings(g)
    
    print(f"Computing raw eigenvalues for g = {g}")

    b.computeHamiltonian(ren=False)
    
    b.computeEigval(sigma=sigma, n=neigs, ren=False)
        
    print("Raw vacuum energy: ", b.vacuumE(ren="raw"))
    print("Raw spectrum: ", b.spectrum(ren="raw"))
    """

if __name__ == "__main__":
    main(sys.argv)
