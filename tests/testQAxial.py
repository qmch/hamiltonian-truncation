# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:51:40 2020

@author: Ian Lim
"""


from statefuncs import State, Basis 
from qedstatefuncs import FermionState, FermionBasis
from phi1234 import Phi1234
from schwinger import Schwinger
from scipy.constants import pi
import numpy as np
import unittest
import time
from numpy.testing import assert_array_equal

def calculateAxialCharge(state):
    
    total = 0
    
    for wn in np.arange(state.nmin, state.nmax+1):
        if wn < 0:
            total -= (state[wn][0]-state[wn][1])
        elif wn > 0:
            total += (state[wn][0]-state[wn][1])
    
    return total

def calculateAxialChargeArray(basis):
    axialCharge = np.empty(basis.size)
    for i in range(basis.size):
        axialCharge[i] = calculateAxialCharge(basis[i])
    #print(axialCharge)
    return axialCharge

class TestQAxial(unittest.TestCase):
    def setUp(self):
        self.schwinger = Schwinger()
        self.schwinger.buildFullBasis(Emax=13., m=0, L=2*pi, bcs="antiperiodic")
        
    
    def testMatrix(self):
        verbose = False
        
        start = time.time()
        
        #print([state.occs for state in a.fullBasis[-1]])
        self.schwinger.buildMatrix()
        
        axialCharge = calculateAxialChargeArray(self.schwinger.fullBasis)
        #print(axialCharge)
        print(f"Basis size: {axialCharge.size}")
        potential = self.schwinger.potential.M.toarray()
        
        for i in np.arange(axialCharge.size):
            for j in np.arange(axialCharge.size):
                if axialCharge[i] != axialCharge[j]:
                    assert(potential[i,j]==0)
        
        #print(f"Runtime: {time.time()-start}")
        
        if verbose:
            print("Free Hamiltonian:")
            print(self.schwinger.h0.M.toarray())
            # we could check that the diagonal elements are the expected energies
            
            #order zero is just the mass term, which comes out to 2*pi here.
            print("Order zero potential matrix:")
            print(self.schwinger.potential.M.toarray())
        
    # def testAxialCharge(self):
    #     axialCharge = calculateAxialChargeArray(self.schwinger.fullBasis)
        
        
    
if __name__ == '__main__':
    unittest.main()