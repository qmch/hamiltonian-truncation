# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:19:10 2020

@author: Ian Lim
"""

import numpy as np
import scipy
from scipy import sqrt, pi
from operator import attrgetter
import math
from statefuncs import omega, k
from statefuncs import State, Basis

class FermionState():
    
    def __init__(self, leftMoverOccs, rightMoverOccs, nmax,
                 L=None, m=None, fast=False, checkAtRest=True,
                 checkChargeNeutral=True):
        """ 
        Args:
            rightMoverOccs: occupation number list
            leftMoverOccs: occupation number list
            nmax (int): wave number of the last element in occs
            fast (bool): a flag for when occs and nmax are all that are needed
                (see transformState in oscillators.py)
            checkAtRest (bool): a flag to check if the total momentum is zero
        
        For instance,
        State([1,0,1],nmax=1) is a state with one excitation in the n=-1 mode
        and one in the n=+1 mode.
        State([1,0,1],nmax=2), however, is a state with one excitation in the
        n=0 mode and one in the n=+2 mode.
        """
        #assert m >= 0, "Negative or zero mass"
        #assert L > 0, "Circumference must be positive"
        assert (len(leftMoverOccs) == len(rightMoverOccs)),\
            "Occupation number lists should match in length"
            
        assert np.all(np.less_equal(leftMoverOccs, 1))\
            and np.all(np.less_equal(rightMoverOccs, 1)),\
            "Pauli exclusion violated"
        
        self.leftMoverOccs = np.array(leftMoverOccs)
        self.rightMoverOccs = np.array(rightMoverOccs)
        self.occs = np.transpose(np.vstack((self.leftMoverOccs,
                                            self.rightMoverOccs)))
        
        self.size = len(self.occs)
        self.nmax = nmax
        self.nmin = self.nmax - self.size + 1
        self.fast = fast
        
        if fast == True:
            return
        
        wavenum = np.arange(self.nmin, self.nmax+1)
        self.totalWN = (wavenum*np.transpose(self.occs)).sum()

        self.netCharge = self.leftMoverOccs.sum() - self.rightMoverOccs.sum()

        if checkAtRest:
            if self.totalWN != 0:            
                raise ValueError("State not at rest")
                
        if checkChargeNeutral:
            if self.netCharge != 0:
                raise ValueError("State not charge-neutral")

        self.__parityEigenstate = (self.size == 2*self.nmax + 1
                                   and np.array_equal(self.occs[::-1],self.occs))
        self.__chargeNeutral = self.netCharge == 0
        
        self.L = L
        self.m = m
        
        energies = omega(wavenum,L,m)
        self.energy = (energies*np.transpose(self.occs)).sum()
        self.momentum = (2.*pi/self.L)*self.totalWN
    
    def __eq__(self, other):
        return np.array_equal(self.occs,other.occs) or np.array_equal(self.occs,other.occs[::-1])

    def __getitem__(self, wn):
        """ Returns the occupation numbers corresponding to a wave number"""
        return self.occs[wn - self.nmin]

    def __hash__(self):
        """Needed for construction of state lookup dictionaries"""
        return hash(tuple(np.reshape(self.occs,2*len(self.occs))))
    
    def isParityEigenstate(self):
        """ Returns True if the Fock space state is a P-parity eigenstate """
        return self.__parityEigenstate

    def isNeutral(self):
        return self.__chargeNeutral

    def __repr__(self):
        return str(self.occs)

    def __setitem__(self, wn, n):
        """ Sets the occupation number corresponding to a wave number """
        if self.fast==False:
            self.energy += ((n-self[wn])*omega(wn,self.L,self.m)).sum()
            self.totalWN += ((n-self[wn])*wn).sum()
            self.momentum = (2.*pi/self.L)*self.totalWN
            #should we update parity eigenstate too? probably - IL
        
        self.occs[wn-self.nmin] = n
    
    def parityReversed(self):
        """ Reverse under P parity """
        if not self.size == 2*self.nmax+1:
            raise ValueError("attempt to reverse asymmetric occupation list")
        return FermionState(self.occs[::-1,0],self.occs[::-1,1],self.nmax,L=self.L,m=self.m)

class FermionBasis(Basis):
    """ Generic list of fermionic basis elements sorted in energy. """
    
    def __init__(self, L, Emax, m, K, nmax=None):
        """ nmax: if not None, forces the state vectors to have length 2nmax+1
            K: field parity (+1 or -1)
        """
        self.L = L
        self.Emax = Emax
        self.m = m
        self.K = K
        
        if nmax == None:
            self.nmax = int(math.floor(sqrt((Emax/2.)**2.-m**2.)*self.L/(2.*pi)))
        else:
            self.nmax=nmax
        
        self.stateList = sorted(self.__buildBasis(), key=attrgetter('energy'))
        # Collection of Fock space states, possibly sorted in energy

        self.reversedStateList = [state.parityReversed() for state in self.stateList]
        # P-parity reversed collection of Fock-space states

        #make a dictionary of states for use in lookup()
        self.statePos = { state : i for i, state in enumerate(self.stateList) }
        self.reversedStatePos = { state : i for i, state in enumerate(self.reversedStateList) }

        self.size = len(self.stateList)
    
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
        kmax = max(0., scipy.sqrt((self.Emax/2.)**2.-self.m**2.))
                
        # the max occupation number of the n=1 mode is either kmax divided 
        # by the momentum at n=1 or Emax/omega, whichever is less
        '''
        if (kmax / k(1,self.L) < 1) or (self.Emax/omega(1,self.L,self.m) < 1):
            maxN1 = 0
        else:
            maxN1 = 1
        '''
        maxN1 = min([math.floor(kmax/k(1,self.L)),
                     math.floor(self.Emax/omega(1,self.L,self.m)),
                     2])
        
        if maxN1 <= 0:
            nextOccs = [[0,0]]
        elif maxN1 == 1:
            nextOccs = [[0,0],[0,1],[1,0]]
        else:
            nextOccs = [[0,0],[0,1],[1,0],[1,1]]
        
        RMlist0 = [FermionState([occs[0]],[occs[1]],1,L=self.L,m=self.m,checkAtRest=False,
                         checkChargeNeutral=False) for occs in nextOccs]
        # seed list of RM states,all possible n=1 mode occupation numbers
        
        #print(RMlist0)
        
        for n in range(2,self.nmax+1): #go over all other modes
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
                    #print(longerstate)
                    #longerstate = numpy.append(longerstate,N)
                    RMlist1.append(FermionState(longerstate[:,0],
                                                longerstate[:,1],
                                                nmax=len(longerstate),
                                                L=self.L,m=self.m,
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
        self.__RMdivided = [[] for ntot in range(self.__nRMmax+1)] #initialize list of lists
        for RMstate in self.__RMlist: #go over RMstates and append them to corresponding sublists
            self.__RMdivided[RMstate.totalWN].append(RMstate)
        
        #now sort each sublist in energy
        for RMsublist in self.__RMdivided:
            RMsublist.sort(key=attrgetter('energy'))
        
    
    # finally function which builds the basis        
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
                    
                    maxN0 = min(int(math.floor(deltaE/self.m)),2)
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
                        state = FermionState(newOccs[:,0],newOccs[:,1],
                                             nmax=self.nmax,L=self.L,m=self.m,
                                             checkAtRest=True,
                                             checkChargeNeutral=False)
                        if state.isNeutral():
                            statelist.append(state)

        return statelist

'''
class FermionMatrix():
    def __init__(self, matrix):
        """
        Parameters
        ----------
        matrix : 2D NumPy array (2x2)
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.matrix = matrix
    
    def __mul__(self, other):
        """ Multiplication of matrix with matrix or number"""
        if isinstance(other, FermionMatrix):
            return self.matrix * other.matrix
        elif isinstance(other, FermionState):
            newLeftOccs = (self.matrix[0,0] * other.leftMoverOccs
                + self.matrix[0,1] * other.rightMoverOccs)
            newRightOccs = (self.matrix[1,0] * other.leftMoverOccs
                + self.matrix[1,1] * other.rightMoverOccs)
            return FermionState(newLeftOccs,newRightOccs,nmax = other.nmax,
                               L = other.L, m = other.m)
        else:
            return self.matrix * other
        
    def __rmul__(self, other):
        """ Define multiplication from the left in the most sensible way """
        return other * self.matrix
    
    def __repr__(self):
        return str(self.matrix)
'''