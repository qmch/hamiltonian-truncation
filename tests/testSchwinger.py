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

class TestSchwinger(unittest.TestCase):
    def setUp(self):
        self.schwinger = Schwinger()
        self.schwinger.buildFullBasis(Emax=3., m=0, L=2*pi)
    
    def testBasisElements(self):
        #print(self.schwinger.fullBasis)
        
        expectedOccs = [([0,0,0],[0,0,0]), ([0,1,0],[0,1,0]),
                          ([1,0,0],[0,0,1]),([1,1,0],[0,1,1]),
                          ([0,0,1],[1,0,0]),([0,1,1],[1,1,0])]
        
        for i, state in enumerate(self.schwinger.fullBasis):
            assert_array_equal(state.particleOccs,expectedOccs[i][0])
            assert_array_equal(state.antiparticleOccs,expectedOccs[i][1])
    
    def testGenerateOperators(self):
        #print(self.schwinger.fullBasis)
        ops = self.schwinger.generateOperators()
        #print(ops)
        
    
    def testMatrix(self):
        verbose = False
        if verbose:
            print("Beginning Schwinger matrix test")
        
        start = time.time()        
        
        #print(a.fullBasis)
        
        #self.assertEqual(len(a.fullBasis[1]),len(expected_basis))
        
        #for index, occs in enumerate(expected_basis):
        #    self.assertEqual(a.fullBasis[1][index].occs, occs)
        
        #ops = a.generateOperators()
        """
        for key in ops.keys():
            for op in ops[key]:
                print(f"Coeff:{op.coeff}, ops:{op}")
        """
        
        #print([state.occs for state in a.fullBasis[-1]])
        self.schwinger.buildMatrix()
        
        print(f"Runtime: {time.time()-start}")
        
        if verbose:
            print("Free Hamiltonian:")
            print(self.schwinger.h0.M.toarray())
            # we could check that the diagonal elements are the expected energies
            
            #order zero is just the mass term, which comes out to 2*pi here.
            print("Order zero potential matrix:")
            print(self.schwinger.potential.M.toarray())
        
    # def testAxialCharge(self):
    #     for n in range(6):
    #         print(calculateAxialCharge(self.schwinger.fullBasis[n]))
    
if __name__ == '__main__':
    unittest.main()