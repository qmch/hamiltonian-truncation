# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:39:07 2020

@author: Ian Lim
"""

from statefuncs import State, Basis 
from phi1234 import Phi1234
from scipy.constants import pi
import numpy as np
import unittest

class TestBasis(unittest.TestCase):
    
    def setUp(self):
        self.Emax = 10.
        self.basis = Basis(L=2*pi, Emax=self.Emax, m=4., K=1)

    def test_nmax(self):
        
        # nmax is computed as
        # nmax = int(math.floor(sqrt((Emax/2.)**2.-m**2.)*self.L/(2.*pi)))
        # with the values above, nmax = 3
        self.assertEqual(self.basis.nmax,3)
    
    def test_basis_elements(self):

        expected_basis = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 0, 0, 1]]
        # note there is no [0, 0, 0, 1, 0, 0, 0] state because such a state
        # has k-parity -1
        
        self.assertEqual(len(self.basis),len(expected_basis))
        
        for index, occs in enumerate(expected_basis):
            self.assertEqual(self.basis[index].occs, occs)
    
    def test_basis_sorted(self):
        
        laststate = None
        for state in self.basis:
            self.assertLessEqual(state.energy, self.Emax)
            if laststate:
                self.assertLessEqual(laststate.energy, state.energy)
            laststate = state

'''
#test the length of a large basis whose length is known
class TestLargeBasis(unittest.TestCase):
    
    def setUp(self):
        self.Emax = 20.
        self.basisk1 = Basis(L=10, Emax=20, m=1., K=1)
        self.basisk2 = Basis(L=10, Emax=20, m=1., K=-1)
    
    def test_length(self):
        self.assertEqual(len(self.basisk1), 12870)
        self.assertEqual(len(self.basisk2), 12801)
        # n.b. these numbers come from footnote 14 in 
        # https://arxiv.org/pdf/1412.3460.pdf
'''  
    
if __name__ == '__main__':
    unittest.main()