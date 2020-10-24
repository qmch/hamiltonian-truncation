

from statefuncs import State, Basis 
from phi1234 import Phi1234
from scipy.constants import pi
import numpy as np
from numpy.testing import assert_array_equal
from qedstatefuncs import FermionBasis, FermionState
from qedops import FermionOperator
import unittest

class TestHalfIntegerNmax(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.nmax = 0.5
        self.m = 0.
        
        #initialize a FermionState with half-integer nmax=0.5
        self.state = FermionState([1,1],[1,1],self.nmax,self.L,self.m)
        self.operator = FermionOperator(clist=[],dlist=[],
                                        anticlist=[],antidlist=[0.5],
                                        L=self.L,m=self.m,normed=True)
    
    def testGetter(self):
        assert_array_equal(self.state[-0.5],[1,1])
        assert_array_equal(self.state[0.5],[1,1])
    
    def testOperator(self):
        
        #test that adding a particle to a filled state annihilates it
        for c1 in (-0.5,0.5):
            operator = FermionOperator([c1],[],[],[],self.L,self.m)
            self.assertEqual(operator._transformState(self.state),
                             (0,None))
            
        for c2 in (-0.5,0.5):
            operator = FermionOperator([],[],[c2],[],self.L,self.m)
            self.assertEqual(operator._transformState(self.state),
                             (0,None))
        
        op1 = FermionOperator([],[-0.5],[],[],self.L,self.m,normed=True)
        outState1 = FermionState([0,1],[1,1],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        n, newState = op1._transformState(self.state, returnCoeff=True)
        self.assertEqual(n,1)
        self.assertEqual(newState, outState1)
        
        op2 = FermionOperator([],[0.5],[],[],self.L,self.m,normed=True)
        outState2 = FermionState([1,0],[1,1],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        n, newState = op2._transformState(self.state, returnCoeff=True)
        self.assertEqual(n,-1)
        self.assertEqual(newState, outState2)
        
        op3 = FermionOperator([],[],[],[-0.5],self.L,self.m,normed=True)
        outState3 = FermionState([1,1],[0,1],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        n, newState = op3._transformState(self.state, returnCoeff=True)
        self.assertEqual(n,1)
        self.assertEqual(newState, outState3)
        
        op4 = FermionOperator([],[],[],[0.5],self.L,self.m,normed=True)
        outState4 = FermionState([1,1],[1,0],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        n, newState = op4._transformState(self.state, returnCoeff=True)
        self.assertEqual(n,-1)
        self.assertEqual(newState, outState4)