######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import phi1234
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
    
    a = phi1234.Phi1234()

    #build the full basis with both values of k (parity)
    a.buildFullBasis(k=1, Emax=Emax, L=L, m=m)
    a.buildFullBasis(k=-1, Emax=Emax, L=L, m=m)

    print("K=1 basis size :", a.fullBasis[1].size)
    print("K=-1 basis size :", a.fullBasis[-1].size)

    #set the file name for saving the generated matrix
    fstr = "Emax="+str(a.fullBasis[1].Emax)+"_L="+str(a.L)

    a.buildMatrix()
    print("Runtime:",time.time()-startTime)
    #a.saveMatrix(fstr)

if __name__ == "__main__":
    main(sys.argv)
