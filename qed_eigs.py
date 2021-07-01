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
        print("python qed_genMatrix.py <fname> <g> <Emax>")
        sys.exit(-1)
    
    
    print("Beginning eigenvalue calculation.")
    startTime = time.time()
    
    fname = argv[1]
    g = float(argv[2])
    Emax = float(argv[3])

    # Hardcoded parameters
    m=1.
    sigma = -30.
    neigs = 3

    b = schwinger.Schwinger()
    b.loadMatrix(fname)

    b.buildBasis(Emax=Emax)
    
    print(f"full basis size = {b.fullBasis.size}")
    print(f"basis size = {b.basis.size}")
    
    b.setcouplings(g)
    
    print(f"Computing raw eigenvalues for g = {g}")

    b.computeHamiltonian(ren=False)
    
    print(f"{b.H.toarray()}")
    
    b.computeEigval(sigma=sigma, n=neigs, ren=False)
        
    print("Raw vacuum energy: ", b.vacuumE(ren="raw"))
    print("Raw spectrum: ", b.spectrum(ren="raw"))

if __name__ == "__main__":
    main(sys.argv)
