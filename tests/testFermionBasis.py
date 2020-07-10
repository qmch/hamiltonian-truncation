

from statefuncs import State, Basis 
from phi1234 import Phi1234
from scipy.constants import pi
import numpy as np
from qedstatefuncs import FermionBasis
import unittest

class TestMasslessFermionBasis(unittest.TestCase):
    def setUp(self):
        self.Emax = 5.
        self.fermionBasis = FermionBasis(L=2*pi, Emax=self.Emax, m=0., K=1)
    
    def testNeutral(self):
        for state in self.fermionBasis:
            self.assertTrue(state.isNeutral())

    def testAtRest(self):
        for state in self.fermionBasis:
            self.assertEqual(state.momentum, 0.)
    
    def testSortedEnergy(self):
        lastEnergy = 0
        for state in self.fermionBasis:
            self.assertLessEqual(lastEnergy, state.energy)
            lastEnergy = state.energy

class TestMassiveFermionBasis(unittest.TestCase):
    def setUp(self):
        self.Emax = 10.
        self.fermionBasis = FermionBasis(L=2*pi, Emax=self.Emax, m=4., K=1)
    
    def testNeutral(self):
        for state in self.fermionBasis:
            self.assertTrue(state.isNeutral())

    def testAtRest(self):
        for state in self.fermionBasis:
            self.assertEqual(state.momentum, 0.)
    
    def testSortedEnergy(self):
        lastEnergy = 0
        for state in self.fermionBasis:
            self.assertLessEqual(lastEnergy, state.energy)
            lastEnergy = state.energy
