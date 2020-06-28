# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:19:10 2020

@author: Ian Lim
"""

import numpy
import scipy
from scipy import sqrt, pi
from operator import attrgetter
import math
from statefuncs import omega, k

class SpinorState():
    
    def __init__(self, leftMoverOccs, rightMoverOccs, nmax,
                 L=None, m=None, fast=False, checkAtRest=True):
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
            
        assert numpy.all(leftMoverOccs <= 1) and numpy.all(rightMoverOccs <=1),\
            "Pauli exclusion violated"
        
        self.leftMoverOccs = leftMoverOccs
        self.rightMoverOccs = rightMoverOccs
        self.occs = numpy.vstack((leftMoverOccs, rightMoverOccs))
        
        self.size = len(self.leftMoverOccs)
        self.nmax = nmax
        self.nmin = self.nmax - self.size + 1
        self.fast = fast
        
        if fast == True:
            return
        
        wavenum = numpy.arange(self.nmin, self.nmax+1)
        self.totalWN = (wavenum*self.occs).sum()

        if checkAtRest:
            if self.totalWN != 0:            
                raise ValueError("State not at rest")

        if self.size == 2*self.nmax+1 and numpy.array_equal(self.occs[:,::-1],self.occs):
            self.__parityEigenstate = True
        else:
            self.__parityEigenstate = False        
            
        self.L = L
        self.m = m
        
        energies = omega(wavenum,L,m)
        self.energy = (energies*self.occs).sum()
        self.momentum = (2.*pi/self.L)*self.totalWN
    
    def __eq__(self, other):
        return numpy.array_equal(self.occs,other.occs) or numpy.array_equal(self.occs,other.occs[::-1])

    def __getitem__(self, wn):
        """ Returns the occupation number corresponding to a wave number"""
        return self.occs[:, wn - self.nmin]

    def __hash__(self):
        """Needed for construction of state lookup dictionaries"""
        return hash(tuple(self.occs))
    
    def isParityEigenstate(self):
        """ Returns True if the Fock space state is a P-parity eigenstate """
        return self.__parityEigenstate        

    def __repr__(self):
        return str(self.occs)

    #needs to be updated for the new version of occs
    def __setitem__(self, wn, n):
        """ Sets the occupation number corresponding to a wave number """
        if self.fast==False:
            self.energy += (n-self[wn])*omega(wn,self.L,self.m)
            self.totalWN += (n-self[wn])*wn
            self.momentum = (2.*pi/self.L)*self.totalWN
        
        self.occs[wn+self.size-self.nmax-1] = n
    
    def parityReversed(self):
        """ Reverse under P parity """
        if not self.size == 2*self.nmax+1:
            raise ValueError("attempt to reverse asymmetric occupation list")
        return SpinorState(self.occs[0,::-1],self.occs[1,::-1],self.nmax,L=self.L,m=self.m)    
