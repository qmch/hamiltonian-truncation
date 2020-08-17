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

class TestSchwinger(unittest.TestCase):    
    def testMatrix(self):
        
        verbose = False
        
        start = time.time()
        
        a = Schwinger()
        a.buildFullBasis(Emax=4., m=0, L=2*pi)
        
        expected_basis = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 0, 0, 1]]
        
        #print(a.fullBasis)
        
        #self.assertEqual(len(a.fullBasis[1]),len(expected_basis))
        
        #for index, occs in enumerate(expected_basis):
        #    self.assertEqual(a.fullBasis[1][index].occs, occs)
        
        ops = a.generateOperators()
        """
        for key in ops.keys():
            for op in ops[key]:
                print(f"Coeff:{op.coeff}, ops:{op}")
        """
        
        #print([state.occs for state in a.fullBasis[-1]])
        a.buildMatrix()
        
        print(f"Runtime: {time.time()-start}")
        
        if verbose:
            print("Free Hamiltonian:")
            print(a.h0[1].M.toarray())
            # we could check that the diagonal elements are the expected energies
            
            #order zero is just the mass term, which comes out to 2*pi here.
            print("Order zero potential matrix:")
            print(a.potential[1][0].M.toarray())
            
            print("Order phi^2 potential matrix:")
            print(a.potential[1][2].M.toarray())
            
            print("Order phi^4 potential matrix:")
            print(a.potential[1][4].M.toarray())
    
if __name__ == '__main__':
    unittest.main()