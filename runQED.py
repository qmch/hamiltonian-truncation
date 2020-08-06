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

def main(argv):
    
    #if there are too few arguments, print the right syntax and exit
    if len(argv) < 3:
        print("python genMatrix.py <L> <Emax>")
        sys.exit(-1)
    
    print("Beginning execution.")
    startTime = time.time()
    #circle circumference (p.5)
    L = float(argv[1])
    #maximum energy (compare Eq. 2.26)
    Emax = float(argv[2])
    #mass
    m = 1.
    
    a = schwinger.Schwinger()
    
    a.buildFullBasis(L, m, Emax)

    print(f"Basis size : {a.fullBasis.size}")

    #set the file name for saving the generated matrix
    fstr = "Emax="+str(a.fullBasis.Emax)+"_L="+str(a.L)

    a.buildMatrix()
    #print(a.h0)
    #print(a.fullBasis)
    print("Runtime:",time.time()-startTime)
    #a.saveMatrix(fstr)

if __name__ == "__main__":
    main(sys.argv)
