######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import phi1234
import renorm
import sys
import scipy
import time

def main(argv):
    if len(argv) < 4:
        print(argv[0], "<fname> <g> <Emax>")
        return -1

    print("Beginning execution.")
    initialTime = time.time()
    
    fname = argv[1]
    g = float(argv[2])
    Emax = float(argv[3])

    # Hardcoded parameters
    m=1.
    sigma = -30.
    neigs = 3

    a = phi1234.Phi1234()
    a.loadMatrix(fname)

    a.buildBasis(k=1, Emax=Emax)
    a.buildBasis(k=-1, Emax=Emax)
    
    print('K=1 full basis size = ',  a.fullBasis[1].size)
    print('K=-1 full basis size = ',  a.fullBasis[-1].size)
    print('K=1 basis size = ',  a.basis[1].size)
    print('K=-1 basis size = ',  a.basis[-1].size)
    
    a.setcouplings(g4=g)

    print("Computing raw eigenvalues for g4 = ", g)

    a.computeHamiltonian(k=1, ren=False)
    a.computeHamiltonian(k=-1, ren=False)

    a.computeEigval(k=1, sigma=sigma, n=neigs, ren=False)
    a.computeEigval(k=-1, sigma=sigma, n=neigs, ren=False)
        
    print("Raw vacuum energy: ", a.vacuumE(ren="raw"))
    print("K=1 Raw spectrum: ", a.spectrum(k=1, ren="raw"))
    print("K=-1 Raw spectrum: ", a.spectrum(k=-1, ren="raw"))
        
    a.renlocal(Er=a.vacuumE(ren="raw"))
        
    print("Computing renormalized eigenvalues for g0r,g2r,g4r = ", a.g0r,a.g2r,a.g4r)
        
    a.computeHamiltonian(k=1, ren=True)
    a.computeHamiltonian(k=-1, ren=True)

    a.computeEigval(k=1, sigma=sigma, n=neigs, ren=True, corr=True)
    a.computeEigval(k=-1, sigma=sigma, n=neigs, ren=True, corr=True)

    print("Renlocal vacuum energy: ", a.vacuumE(ren="renlocal"))
    print("K=1 renlocal spectrum: ", a.spectrum(k=1, ren="renlocal"))
    print("K=-1 renlocal spectrum: ", a.spectrum(k=-1, ren="renlocal"))
    
    print("Rensubl vacuum energy: ", a.vacuumE(ren="rensubl"))
    print("K=1 rensubl spectrum: ", a.spectrum(k=1, ren="rensubl"))
    print("K=-1 rensubl spectrum: ", a.spectrum(k=-1, ren="rensubl"))
        
    print("Total runtime: ",time.time()-initialTime)

if __name__ == "__main__":
    main(sys.argv)
