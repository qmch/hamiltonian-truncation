    

from scipy.constants import pi
import numpy as np
from dressedstatefuncs import DressedFermionState
from statefuncs import omega
import unittest

class TestDressedStateMethods(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.nmax = 1
        self.m = 0.
        
        self.Emax = 5.
        
        #create two states, one with the zero mode in the ground state
        #and one with the zero mode in the first excited state
        #third argument is azeromode
        
        
        #and two more with particles/antiparticles
        # RMstate1 = DressedFermionState([0,0,1],[0,0,0],1, 
        #                             self.nmax,self.L,self.m,
        #                      checkAtRest=False,checkChargeNeutral=False)
        # RMstate2 = DressedFermionState([0,0,0],[0,0,1],1, 
        #                             self.nmax,self.L,self.m,
        #                      checkAtRest=False,checkChargeNeutral=False)
    
    def testSetGet(self):
        state1 = DressedFermionState([0,0,0],[0,0,0],0, 
                                    self.nmax,self.L,self.m,
                             checkAtRest=False,checkChargeNeutral=False)
        state2 = DressedFermionState([0,0,0],[0,0,0],1, 
                                    self.nmax,self.L,self.m,
                             checkAtRest=False,checkChargeNeutral=False)
        
        omega0 = 1/np.sqrt(pi)
        
        self.assertEqual(state1.getAZeroMode(),0)
        self.assertEqual(state2.getAZeroMode(),1)
        
        self.assertEqual(state1.energy,0)
        self.assertEqual(state2.energy,omega0)
        
        state1.setAZeroMode(1)
        self.assertEqual(state1,state2)
        
        #check that the energy has updated correctly
        self.assertAlmostEqual(state1.energy,omega0)
        
        state1.setAZeroMode(2)
        self.assertAlmostEqual(state1.energy,2*omega0)
        
        #reset the zero mode to the 1-state
        state1.setAZeroMode(1)
        #add a fermion with k=1
        state1[1] = [1,0]
        self.assertAlmostEqual(state1.energy,1+omega0)
        
        #add an antifermion with k=-1
        state1[-1] = [0,1]
        self.assertAlmostEqual(state1.energy,2+omega0)
        
        #print(state1)
