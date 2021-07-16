    

from scipy.constants import pi
import numpy as np
from dressedstatefuncs import DressedFermionState, DressedFermionBasis
from dressedstatefuncs import ZeroModeRaisingOperator, ZeroModeLoweringOperator
from statefuncs import omega, NotInBasis
from numpy import sqrt
import unittest

class TestDressedStateMethods(unittest.TestCase):
    def setUp(self):
        self.L = 2*pi
        self.nmax = 1
        self.m = 0.
        
        self.Emax = 2.
        
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

    def testBasis(self):
        # periodic_basis = DressedFermionBasis(self.L, self.Emax,
        #                                      self.m, bcs="periodic")
        
        antiperiodic_basis = DressedFermionBasis(self.L, self.Emax,
                                                 self.m, bcs="antiperiodic")
        
        self.assertEqual(antiperiodic_basis.size,9)
        #optional: can print basis elements
        #expected output: 
        # [(Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 0),
        # (Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 1),
        # (Particle occs: [1 0], antiparticle occs: [0 1], zero mode: 0),
        # (Particle occs: [0 1], antiparticle occs: [1 0], zero mode: 0),
        # (Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 2),
        # (Particle occs: [1 0], antiparticle occs: [0 1], zero mode: 1),
        # (Particle occs: [0 1], antiparticle occs: [1 0], zero mode: 1),
        # (Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 3),
        # (Particle occs: [1 1], antiparticle occs: [1 1], zero mode: 0)]
        
        # energies:
        # 0.0
        # 0.5641895835477563
        # 1.0
        # 1.0
        # 1.1283791670955126
        # 1.5641895835477562
        # 1.5641895835477562
        # 1.692568750643269
        # 2.0
        
class TestRaisingLoweringOperators(unittest.TestCase):
    def testTransformState(self):
        raising_op = ZeroModeRaisingOperator()
        lowering_op = ZeroModeLoweringOperator()
        transformedstate = DressedFermionState([0,0],[0,0],1,0.5,L=2*pi,m=0)
        
        mybasis = DressedFermionBasis(2*pi, 1.5, 0, bcs="antiperiodic")
        #print(mybasis)
        # expected output:
        # [(Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 0),
        #  (Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 1),
        #  (Particle occs: [1 0], antiparticle occs: [0 1], zero mode: 0),
        #  (Particle occs: [0 1], antiparticle occs: [1 0], zero mode: 0),
        #  (Particle occs: [0 0], antiparticle occs: [0 0], zero mode: 2)]
        
        n, newstate = raising_op._transformState(mybasis[0])
        
        self.assertEqual(n, 1.0)
        self.assertEqual(newstate,transformedstate)
        
        n, newstate = lowering_op._transformState(newstate)
        
        self.assertEqual(n,1.0)
        self.assertEqual(newstate,mybasis[0])

        n, newstate = lowering_op._transformState(mybasis[0])
        
        self.assertEqual(n, 0)
        self.assertEqual(newstate, None)
        
        n, newstate = lowering_op._transformState(mybasis[4])
        
        self.assertAlmostEqual(n,sqrt(2))
        self.assertEqual(newstate,mybasis[1])
        
        n, index = raising_op.apply2(mybasis, mybasis[0])
        
        self.assertEqual(n,1.0)
        self.assertEqual(index,1)
        
        n, index = raising_op.apply2(mybasis, mybasis[1])
        
        self.assertEqual(n,sqrt(2))
        self.assertEqual(index,4)
        
        #raise a state out of the basis
        self.assertRaises(NotInBasis, raising_op.apply2, mybasis, mybasis[4])
        
        #annihilate a state
        n, index = lowering_op.apply2(mybasis, mybasis[0])
        
        self.assertEqual(n,0)
        self.assertEqual(index,None)
        
        
        