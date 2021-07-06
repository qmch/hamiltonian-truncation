# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:19:10 2020

@author: Ian Lim
"""

import numpy as np
import scipy
from numpy import sqrt, pi
from operator import attrgetter
import math
from statefuncs import omega, k
from statefuncs import State, Basis, NotInBasis
from qedstatefuncs import FermionState, FermionBasis


class DressedFermionState(FermionState):
    
    def __init__(self, particleOccs, antiparticleOccs, zeromode, nmax,
                 L=None, m=None, fast=False, checkAtRest=True,
                 checkChargeNeutral=True, e_var=1):
        """ 
        Args:
            antiparticleOccs: occupation number list
            particleOccs: occupation number list
            zeromode (int): int specifying eigenstate of the A1 zero mode
            nmax (int): wave number of the last element in occs
            fast (bool): a flag for when occs and nmax are all that are needed
                (see transformState in oscillators.py)
            checkAtRest (bool): a flag to check if the total momentum is zero
            e_var (float): charge (default 1), used in calculating zero mode energy
        
        """
        #assert m >= 0, "Negative or zero mass"
        #assert L > 0, "Circumference must be positive"
        assert zeromode >= 0, "zeromode eigenstate label should be >=0"
        
        self.zeromode = zeromode
        self.omega0 = e_var/sqrt(pi)
        
        FermionState.__init__(self, particleOccs, antiparticleOccs, nmax,
                                            L, m, fast, checkAtRest, checkChargeNeutral)
        
        if fast: return

        self.energy += self.zeromode * self.omega0
    
    # removed behavior checking equality of states up to parity
    def __eq__(self, other):
        return np.array_equal(self.occs,other.occs) and self.zeromode == other.zeromode
    
    def __hash__(self):
        """Needed for construction of state lookup dictionaries"""
        return hash(tuple(np.append(np.reshape(self.occs,2*len(self.occs)),[self.zeromode])))
    
    def getAZeroMode(self):
        return self.zeromode

    def __repr__(self):
        return f"(Particle occs: {self.occs.T[0]}, antiparticle occs: {self.occs.T[1]}, zero mode: {self.zeromode})"
        
    def setAZeroMode(self, n):
        if self.fast==False:
            self.energy += self.omega0*(n-self.zeromode)
        self.zeromode = n    

class DressedFermionBasis(FermionBasis):
    """ List of fermionic basis elements dressed with a zero-mode wavefunction"""
    def __init__(self, L, Emax, m, nmax=None, bcs="periodic", q=1):
        self.q = q
        self.omega0 = q/sqrt(pi)
        """ nmax: if not None, forces the state vectors to have length 2nmax+1
        """
        self.L = L
        self.Emax = Emax
        self.m = m
        
        assert bcs == "periodic" or bcs == "antiperiodic"
        self.bcs = bcs
        
        if nmax == None:
            if bcs == "periodic":
                self.nmax = int(math.floor(sqrt((Emax/2.)**2.-m**2.)*self.L/(2.*pi)))
            elif bcs == "antiperiodic":
                self.nmax = half_floor(sqrt((Emax/2.)**2.-m**2.)*self.L/(2.*pi))
        else:
            self.nmax = nmax
        
        self.stateList = sorted(self.__buildBasis(), key=attrgetter('energy'))
        # Collection of Fock space states, possibly sorted in energy

        self.reversedStateList = [state.parityReversed() for state in self.stateList]
        # P-parity reversed collection of Fock-space states

        #make a dictionary of states for use in lookup()
        self.statePos = { state : i for i, state in enumerate(self.stateList) }
        self.reversedStatePos = { state : i for i, state in enumerate(self.reversedStateList) }

        self.size = len(self.stateList)
        
    def __buildBasis(self):
        """ creates basis of states of total momentum zero and energy <=Emax """
        self.__buildRMlist()
        self.__divideRMlist()
        
        statelist = []

        for nRM,RMsublist in enumerate(self.__RMdivided):
            for i, RMstate in enumerate(RMsublist):
                ERM = RMstate.energy
                for LMstate in RMsublist: # LM part of the state will come from the same sublist. We take the position of LMState to be greater or equal to the position of RMstate
                    #we will just have to reverse it
                    ELM = LMstate.energy
                    deltaE = self.Emax - ERM - ELM
                    if deltaE < 0: #if this happens, we can break since subsequent LMstates have even higherenergy (RMsublist is ordered in energy)
                        break
                    
                    if self.bcs == "antiperiodic":
                        #there is no zero mode with half integer n
                        newOccs = np.array((LMstate.occs[::-1]).tolist()
                                           + RMstate.occs.tolist())
                        state = DressedFermionState(newOccs[:,0],newOccs[:,1],0,
                                             nmax=self.nmax,L=self.L,m=self.m,
                                             checkAtRest=True,
                                             checkChargeNeutral=False)
                        if state.isNeutral():
                            statelist.append(state)
                            addZeroModes(statelist, state.energy, newOccs)
                                
                        continue
                    else:
                        assert self.bcs == "periodic"
                    # for massless excitations we can put the max of 2
                    # excitations in the zero mode.
                    # this is different from the bosonic case where
                    # massless excitations carry no energy and the number
                    # of zero mode excitations is unbounded.
                    # we can manually set this to not add particles to the
                    # zero mode by taking maxN0 = 0.
                    if self.m != 0:
                        maxN0 = min(int(math.floor(deltaE/self.m)),2)
                    else:
                        maxN0 = 2
                    
                    assert maxN0 in range(3)
                    
                    
                    if maxN0 == 0:
                        nextOccsList = [[0,0]]
                    elif maxN0 == 1:
                        nextOccsList = [[0,0],[0,1],[1,0]]
                    else:
                        nextOccsList = [[0,0],[0,1],[1,0],[1,1]]
                    
                    for nextOccs in nextOccsList:
                        newOccs = np.array((LMstate.occs[::-1]).tolist()
                                           + [nextOccs] + RMstate.occs.tolist())
                        #print(newOccs)
                        state = DressedFermionState(newOccs[:,0],newOccs[:,1],0,
                                             nmax=self.nmax,L=self.L,m=self.m,
                                             checkAtRest=True,
                                             checkChargeNeutral=False)
                        if state.isNeutral():
                            statelist.append(state)
                            self.addZeroModes(statelist, state.energy, newOccs)
        return statelist
    
    def __buildRMlist(self):
        """ sets list of all right-moving states with particles of individual wave number 
        <= nmax, total momentum <= Emax/2 and total energy <= Emax
        This function works by first filling in n=1 mode in all possible ways, then n=2 mode
        in all possible ways assuming the occupation of n=1 mode, etc
        
        This is modified for fermionic states. In the fermionic case,
        occupation numbers are zero or one due to Pauli exclusion.
        """
        
        if self.nmax == 0:
            self.__RMlist = [FermionState([],[],nmax=0,L=self.L,m=self.m,
                                          checkAtRest=False,
                                          checkChargeNeutral=False)]
            return
        
        # for zero-momentum states, the maximum value of k is as follows.
        kmax = max(0., np.sqrt((self.Emax/2.)**2.-self.m**2.))
                
        # the max occupation number of the n=1 mode is either kmax divided 
        # by the momentum at n=1 or Emax/omega, whichever is less
        # the 2 here accounts for that we can have a single particle and an 
        # antiparticle in n=1
        if self.bcs == "periodic":
            seedN = 1
        elif self.bcs == "antiperiodic":
            seedN = 0.5
        
        maxN1 = min([math.floor(kmax/k(seedN,self.L)),
                     math.floor(self.Emax/omega(seedN,self.L,self.m)),
                     2])
        
        if maxN1 <= 0:
            nextOccs = [[0,0]]
        elif maxN1 == 1:
            nextOccs = [[0,0],[0,1],[1,0]]
        else:
            nextOccs = [[0,0],[0,1],[1,0],[1,1]]
        
        RMlist0 = [FermionState([occs[0]],[occs[1]],seedN,L=self.L,m=self.m,checkAtRest=False,
                         checkChargeNeutral=False) for occs in nextOccs]
        # seed list of RM states,all possible n=1 mode occupation numbers
        
        
        for n in np.arange(seedN+1,self.nmax+1): #go over all other modes
            RMlist1=[] #we will take states out of RMlist0, augment them and add to RMlist1
            for RMstate in RMlist0: # cycle over all RMstates
                p0 = RMstate.momentum
                e0 = RMstate.energy
                
                # maximal occupation number of mode n given the momentum/energy
                # in all previous modes. The sqrt term accounts for the
                # ground state energy of the overall state, while e0 gives
                # the energy in each of the mode excitations.
                maxNn = min([math.floor((kmax-p0)/k(n,self.L)),
                             math.floor((self.Emax-np.sqrt(self.m**2+p0**2)-e0)/omega(n,self.L,self.m)),
                             2])
                
                if maxNn <= 0:
                    nextOccsList = [[0,0]]
                elif maxNn == 1:
                    nextOccsList = [[0,0],[0,1],[1,0]]
                else:
                    nextOccsList = [[0,0],[0,1],[1,0],[1,1]]
                
                assert maxNn <= 2, f"maxNn was {maxNn}"
                # got to here in edits.
                # should we maybe just write a numpy function to calculate
                # energy and momentum from the occs list?
                # update: i did this. But it would take an extra
                # function call instead of accessing state properties.
                
                #print(f"RMstate occs are {RMstate.occs}")
                for nextOccs in nextOccsList:
                    longerstate = np.append(RMstate.occs,[nextOccs],axis=0)
                    RMlist1.append(FermionState(longerstate[:,0],
                                                longerstate[:,1],
                                                nmax=n,L=self.L,m=self.m,
                                                checkAtRest=False,
                                                checkChargeNeutral=False))
            #RMlist1 created, copy it back to RMlist0
            RMlist0 = RMlist1
        
        self.__RMlist = RMlist0 #save list of RMstates in an internal variable 
    
    def __divideRMlist(self):
        """ divides the list of RMstates into a list of lists, RMdivided,
        so that two states in each list have a fixed total RM wavenumber,
        also each sublist is ordered in energy"""
        
        self.__nRMmax=max([RMstate.totalWN for RMstate in self.__RMlist])
        if self.bcs == "periodic":
            self.__RMdivided = [[] for ntot in range(self.__nRMmax+1)] #initialize list of lists
        elif self.bcs == "antiperiodic":
            self.__RMdivided = [[] for ntot in np.arange(self.__nRMmax*2+2)] #initialize list of lists
        for RMstate in self.__RMlist: #go over RMstates and append them to corresponding sublists
            if self.bcs == "periodic":
                self.__RMdivided[RMstate.totalWN].append(RMstate)
            elif self.bcs == "antiperiodic":
                self.__RMdivided[int(RMstate.totalWN*2)].append(RMstate)
        
        #now sort each sublist in energy
        for RMsublist in self.__RMdivided:
            RMsublist.sort(key=attrgetter('energy'))
    
    def addZeroModes(self, statelist, state_energy, occs):
        #print("start addZeromodes")
        zeromodetemp = 1
        deltaE_zeromode = self.Emax - state_energy
        while (deltaE_zeromode > self.omega0):
            #print(deltaE_zeromode)
            state = DressedFermionState(occs[:,0],occs[:,1],zeromodetemp,
                                             nmax=self.nmax,L=self.L,m=self.m)
            statelist.append(state)
            deltaE_zeromode -= self.omega0
            zeromodetemp += 1
        
        return