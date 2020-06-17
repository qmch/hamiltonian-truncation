# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:57:07 2020

@author: Ian Lim
"""

#import sys
from statefuncs import State, npState
import numpy as np
import unittest

class TestStates(unittest.TestCase):    
    def test_state_setup(self):
        occs = [0,1,0,1,0]
        nmax = 2
        
        #if L or m is not provided, raise an error
        with self.assertRaises(TypeError):
            State(occs,nmax)
        with self.assertRaises(TypeError):
            State(occs,nmax,L=1.)
        with self.assertRaises(TypeError):
            State(occs,nmax,m=1.)
        
        #should raise ValueError: state not at rest when nmax is shifted
        with self.assertRaises(ValueError):
            State(occs,1,m=1.,L=1.)
    
    def test_npstate_setup(self):
        occs = [0,1,0,1,0]
        nmax = 2
        
        with self.assertRaises(TypeError):
            npState(occs,nmax)
        with self.assertRaises(TypeError):
            npState(occs,nmax,L=1.)
        with self.assertRaises(TypeError):
            npState(occs,nmax,m=1.)
    
        #should raise ValueError: state not at rest when nmax is shifted
        with self.assertRaises(ValueError):
            npState(occs,1,m=1.,L=1.)
        
    def test_state_equality(self):
        occs = [2,1,0,1,2]
        nmax = 2
        occsList = [[2,1,0,1,2],[0,0,0,0,0],[0,5,0,5,0]]
        myState = State(occs,nmax, m=2,L=1.)
        
        for occs in occsList:
            for mass in np.arange(5):
                myState = State(occs, nmax, m=mass, L=1.)
                mynpState = npState(occs, nmax, m=mass, L=1.)
                
                self.assertEqual(myState.energy, mynpState.energy)
                self.assertEqual(myState.isParityEigenstate, mynpState.isParityEigenstate)
                self.assertEqual(myState.momentum,mynpState.momentum)
                self.assertEqual(myState.totalWN,mynpState.totalWN)
    
if __name__ == "__main__":
    unittest.main()