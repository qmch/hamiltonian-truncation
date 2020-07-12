

from statefuncs import State, Basis 
from phi1234 import Phi1234
from scipy.constants import pi
import numpy as np
from qedops import uspinor, vspinor
from statefuncs import omega
import unittest

class TestSpinorIdentities(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.m = 1.
        self.n = 5
        self.k = (2.*pi/self.L)*self.n
        self.E = omega(self.n, self.L, self.m)
        
        self.GAMMA0 = np.array([[0,1],[1,0]])
        self.GAMMA1 = np.array([[0,-1],[1,0]])
        self.myUSpinor = uspinor(self.n, self.L, self.m)
        self.myVSpinor = vspinor(self.n, self.L, self.m)
    
    def testUbarU(self):
        ubaru = np.vdot(self.myUSpinor, np.dot(self.GAMMA0,self.myUSpinor))
        self.assertAlmostEqual(ubaru, 2*self.m)
    
    def testUdaggerU(self):
        udaggeru = np.vdot(self.myUSpinor, self.myUSpinor)
        self.assertEqual(udaggeru, 2*self.E)

    def testVbarV(self):
        vbarv = np.vdot(self.myVSpinor, np.dot(self.GAMMA0, self.myVSpinor))
        self.assertAlmostEqual(vbarv, -2*self.m)
        
    def testVdaggerV(self):
        vdaggerv = np.vdot(self.myVSpinor, self.myVSpinor)
        self.assertEqual(vdaggerv, 2*self.E)
        
class TestOrthogonalSpinors(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.m = 1.
        self.n = 5
        self.k = (2.*pi/self.L)*self.n
        self.E = omega(self.n, self.L, self.m)
        
        self.GAMMA0 = np.array([[0,1],[1,0]])
        self.GAMMA1 = np.array([[0,-1],[1,0]])
        self.myUSpinor = uspinor(self.n, self.L, self.m)
        self.myVSpinor = vspinor(self.n, self.L, self.m)
        # make spinors with -p
        self.myUSpinor2 = uspinor(-self.n, self.L, self.m)
        self.myVSpinor2 = vspinor(-self.n, self.L, self.m)
        
    def testUVorthogonal(self):
        ubarv = np.vdot(self.myUSpinor, np.dot(self.GAMMA0,self.myVSpinor))
        self.assertEqual(ubarv, 0)
        
        vbaru = np.vdot(self.myVSpinor, np.dot(self.GAMMA0,self.myUSpinor))
        self.assertEqual(vbaru, 0)
        
        # orthogonality with the opposite momentum spinors
        udaggerv = np.vdot(self.myUSpinor, self.myVSpinor2)
        self.assertEqual(udaggerv, 0)
        
        vdaggeru = np.vdot(self.myVSpinor2, self.myUSpinor)
        self.assertEqual(vdaggeru, 0)