######################################################
# 
# 
#
######################################################

import phi1234
import sys
import scipy
import time
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    '''
    #if there are too few arguments, print the right syntax and exit
    if len(argv) < 3:
        print("python genMatrix.py <L> <Emax>")
        sys.exit(-1)
    '''
    
    print("Beginning execution.")
    startTime = time.time()
    L = 10.
    Emax = 15.
    #mass
    m = 1.
    
    a = phi1234.Phi1234()

    #build the full basis with both values of k (parity)
    a.buildFullBasis(k=1, Emax=Emax, L=L, m=m)
    a.buildFullBasis(k=-1, Emax=Emax, L=L, m=m)

    print("K=1 basis size :", a.fullBasis[1].size)
    print("K=-1 basis size :", a.fullBasis[-1].size)

    #set the file name for saving the generated matrix
    datestring = time.strftime("%Y_%m_%d", time.gmtime())
    fstr = "./phi4data/" + datestring + f"_Emax={a.fullBasis[1].Emax}_L={a.L}"
    print(fstr)
    
    try:
        a.loadMatrix(fstr+".npz")
        print("Matrix loaded from file.")
    except:
        print("No saved matrix found. Building matrix.")
        a.buildMatrix()
        a.saveMatrix(fstr)
    
    print(f"Building/loading matrix took: {time.time()-startTime}s")
    startTime = time.time()

    a.buildBasis(k=1, Emax=Emax)
    a.buildBasis(k=-1, Emax=Emax)
    
    numsamples = 11
    vacuumEnergiesRen = np.zeros(numsamples)
    vacuumEnergiesSubl = np.zeros(numsamples)
    gvalues = np.linspace(0,5,numsamples)
    for i, g in enumerate(gvalues):
        vacuumEnergiesRen[i], vacuumEnergiesSubl[i] = computeVacuumEnergy(a, g)
    
    print(f"Computing eigenvalues took: {time.time()-startTime}s")
    
    #fig, ax =  plt.subplots()
    plt.plot(gvalues, vacuumEnergiesRen, "x-k", label="ren.")
    plt.plot(gvalues, vacuumEnergiesSubl, ".--k", label="subl.")
    plt.xlabel(r'$g$')
    plt.ylabel(r'$\mathcal{E}$')
    plt.title(r'$m=1, L=10, E_\mathrm{max}=15$')
    plt.legend()
    #print("gvalues are", gvalues)
    #print("vacuum energies are", vacuumEnergies)
    plt.show()
    

def computeVacuumEnergy(phi4, g):
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
    neigs = 3

    phi4.setcouplings(g4=g)

    print(f"Computing raw eigenvalues for g4 = {g}")

    phi4.computeHamiltonian(k=1, ren=False)
    phi4.computeHamiltonian(k=-1, ren=False)

    phi4.computeEigval(k=1, sigma=sigma, n=neigs, ren=False)
    phi4.computeEigval(k=-1, sigma=sigma, n=neigs, ren=False)
        
    print("Raw vacuum energy: ", phi4.vacuumE(ren="raw"))
    #print("K=1 Raw spectrum: ", phi4.spectrum(k=1, ren="raw"))
    #print("K=-1 Raw spectrum: ", phi4.spectrum(k=-1, ren="raw"))
        
    phi4.renlocal(Er=phi4.vacuumE(ren="raw"))
        
    print("Computing renormalized eigenvalues for g0r,g2r,g4r = ", phi4.g0r,phi4.g2r,phi4.g4r)
        
    phi4.computeHamiltonian(k=1, ren=True)
    phi4.computeHamiltonian(k=-1, ren=True)

    phi4.computeEigval(k=1, sigma=sigma, n=neigs, ren=True, corr=True)
    phi4.computeEigval(k=-1, sigma=sigma, n=neigs, ren=True, corr=True)

    print("Renlocal vacuum energy: ", phi4.vacuumE(ren="renlocal"))
    #print("K=1 renlocal spectrum: ", phi4.spectrum(k=1, ren="renlocal"))
    #print("K=-1 renlocal spectrum: ", phi4.spectrum(k=-1, ren="renlocal"))
    
    print("Rensubl vacuum energy: ", phi4.vacuumE(ren="rensubl"))
    #print("K=1 rensubl spectrum: ", phi4.spectrum(k=1, ren="rensubl"))
    #print("K=-1 rensubl spectrum: ", phi4.spectrum(k=-1, ren="rensubl"))
    return phi4.vacuumE(ren="renlocal"), phi4.vacuumE(ren="rensubl")
        

if __name__ == "__main__":
    main(sys.argv)
