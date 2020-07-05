

from statefuncs import State, Basis 
from phi1234 import Phi1234
from scipy.constants import pi
import numpy as np
from qedstatefuncs import FermionBasis
import unittest

class TestFermionBasis(unittest.TestCase):
    
    def testBasis(self):
        self.Emax = 20.
        self.basis = Basis(L=2*pi, Emax=self.Emax, m=4., K=1)
        
        self.fermionBasis = FermionBasis(L=2*pi, Emax=self.Emax, m=4., K=1)
        
        for state in self.fermionBasis:
            self.assertTrue(self.basis.lookup(state))