

from scipy.constants import pi
import numpy as np
from qedops import uspinor, vspinor, FermionOperator
from qedstatefuncs import FermionState
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
        
class TestMasslessSpinors(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.m = 0.
        self.n = 5
        self.k = (2.*pi/self.L)*self.n
        self.E = omega(self.n, self.L, self.m)
        
        self.myUSpinor = uspinor(self.n, self.L, self.m)
        self.myVSpinor = vspinor(self.n, self.L, self.m)
        # make spinors with -p
        self.myUSpinor2 = uspinor(-self.n, self.L, self.m)
        self.myVSpinor2 = vspinor(-self.n, self.L, self.m)
        
        self.myUSpinorNormed = uspinor(self.n, self.L, self.m, normed=True)
        self.myVSpinorNormed = vspinor(self.n, self.L, self.m, normed=True)
        # make spinors with -p
        self.myUSpinor2Normed = uspinor(-self.n, self.L, self.m, normed=True)
        self.myVSpinor2Normed = vspinor(-self.n, self.L, self.m, normed=True)
    
    def testStepFunction(self):
        # test the step function behavior in the massless limit
        
        # the massless u-spinor for positive n should be [sqrt(2E),0]
        self.assertEqual(np.sqrt(2*self.E),self.myUSpinor[0])
        self.assertEqual(0,self.myUSpinor[1])
        
        # the massless v-spinor for positive n should be [sqrt(2E),0]
        self.assertEqual(np.sqrt(2*self.E),self.myVSpinor[0])
        self.assertEqual(0,self.myVSpinor[1])
        
        # the massless u-spinor for positive n should be [0, sqrt(2E)]
        self.assertEqual(0,self.myUSpinor2[0])
        self.assertEqual(np.sqrt(2*self.E),self.myUSpinor2[1])
        
        # the massless u-spinor for positive n should be [0, -sqrt(2E)]
        self.assertEqual(0,self.myVSpinor2[0])
        self.assertEqual(-np.sqrt(2*self.E),self.myVSpinor2[1])
    
    def testStepFunctionNormed(self):
        # test the step function behavior in the massless limit after
        # normalizing by a factor of sqrt(2E)
        self.assertEqual(1.,self.myUSpinorNormed[0])
        self.assertEqual(0,self.myUSpinorNormed[1])
        
        self.assertEqual(1.,self.myVSpinorNormed[0])
        self.assertEqual(0,self.myVSpinorNormed[1])
        
        self.assertEqual(0,self.myUSpinor2Normed[0])
        self.assertEqual(1.,self.myUSpinor2Normed[1])
        
        self.assertEqual(0,self.myVSpinor2Normed[0])
        self.assertEqual(-1.,self.myVSpinor2Normed[1])
        
class TestFermionOperator(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.nmax = 1
        self.m = 0.
        
        self.Emax = 5.
        #self.fermionBasis = FermionBasis(L=2*pi, Emax=self.Emax, m=0.)
        
        self.state = FermionState([1,0,1],[1,0,1],self.nmax,self.L,self.m)
        
        #create an operator that annihilates a particle of momentum 1
        self.operator = FermionOperator(clist=[],dlist=[1],anticlist=[],
                                        antidlist=[],L=self.L,m=self.m,
                                        normed=True, extracoeff=1)
        
    def testApplyOperator(self):
        n, newState = self.operator._transformState(self.state,
                                                    returnCoeff=True)
        #print(np.transpose(newState.occs))
        
        self.assertEqual(n,-1)
        
        #test that adding a particle to a filled state annihilates it
        for c1 in (-1,1):
            operator = FermionOperator([c1],[],[],[],self.L,self.m)
            self.assertEqual(operator._transformState(self.state),
                             (0,None))
        
        for c2 in (-1,1):
            operator = FermionOperator([],[],[c2],[],self.L,self.m)
            self.assertEqual(operator._transformState(self.state),
                             (0,None))
        
        #test that removing a particle from an empty state annihilates it
        #note: we shouldn't really act on the zero mode at all
        #because it is not well-defined for the massless fermion
        operator = FermionOperator([],[0],[],[],self.L,self.m,normed=True)
        self.assertEqual(operator._transformState(self.state), (0,None))
        
        operator = FermionOperator([],[],[],[0],self.L,self.m,normed=True)
        self.assertEqual(operator._transformState(self.state), (0,None))
    
    def testDestructionOperatorSigns(self):
        # test that destruction operators correctly anticommute when applied
        # to states with our convention
        dOperator = FermionOperator(clist=[],dlist=[1],anticlist=[],
                                    antidlist=[],L=self.L,m=self.m,
                                    normed=True, extracoeff=1)
        
        # this state has excitations in the n=+1 and n=-1 modes
        # so there is one trivial anticommutation
        state1 = FermionState([1,0,1],[0,0,0],self.nmax,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        # this state only has an excitation in the +1 mode
        state2 = FermionState([0,0,1],[0,0,0],self.nmax,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        # this state has a particle and an antiparticle
        # so there is one trivial anticommutation with the antiparticle op
        state3 = FermionState([0,0,1],[0,0,1],self.nmax,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        
        outState1 = FermionState([1,0,0],[0,0,0],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        outState2 = FermionState([0,0,0],[0,0,0],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        outState3 = FermionState([0,0,0],[0,0,1],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        
        n, newState = dOperator._transformState(state1, returnCoeff=True)
        self.assertEqual(n,-1)
        self.assertEqual(newState, outState1)
        
        n, newState = dOperator._transformState(state2, returnCoeff=True)
        self.assertEqual(n,1)
        self.assertEqual(newState, outState2)
        
        n, newState = dOperator._transformState(state3, returnCoeff=True)
        self.assertEqual(n,-1)
        self.assertEqual(newState, outState3)
    
    def testCreationOperatorOrdering(self):
        # these operators have the creation operators in a different order
        # so there should be a relative sign in applying these operators
        op1 = FermionOperator([-1,1],[],[],[],self.L,self.m,normed=True)
        op2 = FermionOperator([1,-1],[],[],[],self.L,self.m,normed=True)
        
        # do this also for antiparticle creation operators
        op3 = FermionOperator([],[],[-1,1],[],self.L,self.m,normed=True)
        op4 = FermionOperator([],[],[1,-1],[],self.L,self.m,normed=True)
        
        state = FermionState([0,0,0],[0,0,0],self.nmax,self.L,self.m,
                             checkAtRest=False,checkChargeNeutral=False)
        
        outState = FermionState([1,0,1],[0,0,0],self.nmax,self.L,self.m,
                                checkAtRest=False,checkChargeNeutral=False)
        outState2 = FermionState([0,0,0],[1,0,1],self.nmax,self.L,self.m,
                                 checkAtRest=False,checkChargeNeutral=False)
        
        n, newState = op1._transformState(state, returnCoeff=True)
        self.assertEqual(n,1)
        self.assertEqual(newState,outState)
        
        n, newState = op2._transformState(state, returnCoeff=True)
        self.assertEqual(n,-1)
        self.assertEqual(newState,outState)
        
        n, newState = op3._transformState(state, returnCoeff=True)
        self.assertEqual(n,1)
        self.assertEqual(newState,outState2)
        
        n, newState = op4._transformState(state, returnCoeff=True)
        self.assertEqual(n,-1)
        self.assertEqual(newState,outState2)
    
    def testOrderingParticleAntiparticles(self):
        state1 = FermionState([0,1,0],[0,0,0],2,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        state2 = FermionState([1,1,0],[0,0,0],2,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        state3 = FermionState([1,1,0],[1,0,0],2,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        state4 = FermionState([1,1,0],[1,0,1],2,self.L,self.m,
                              checkAtRest=False,checkChargeNeutral=False)
        
        op1 = FermionOperator([2],[],[],[],self.L,self.m,normed=True)
        
        n, newState = op1._transformState(state1, returnCoeff=True)
        #one anticommutation to get to the right spot
        self.assertEqual(n,-1)
        
        #two anticommutations
        n, newState = op1._transformState(state2, returnCoeff=True)
        self.assertEqual(n,1)
        
        #three anticommutations
        n, newState = op1._transformState(state3, returnCoeff=True)
        self.assertEqual(n,-1)
        
        #four anticommutations ah ah ah
        n, newState = op1._transformState(state4, returnCoeff=True)
        self.assertEqual(n,1)