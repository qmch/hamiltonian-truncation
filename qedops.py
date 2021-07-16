######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import scipy
import numpy as np
from numpy import pi, sqrt, product
from operator import attrgetter
from statefuncs import omega, State, NotInBasis
from qedstatefuncs import FermionState
from dressedstatefuncs import DressedFermionState

tol = 0.0001

def uspinor(n,L,m,normed=False):
    if m == 0 and n == 0:
        return np.array([[1/sqrt(2)],[1/sqrt(2)]])
    k = (2.*pi/L)*n
    energy = omega(n,L,m)
    if normed:
        return np.array([[sqrt(energy-k)/sqrt(2*energy)],
                         [sqrt(energy+k)/sqrt(2*energy)]])
    return np.array([[sqrt(energy-k)],[sqrt(energy+k)]])

def vspinor(n,L,m,normed=False):
    if m == 0 and n == 0:
        return np.array([[1/sqrt(2)],[-1/sqrt(2)]])
    k = (2.*pi/L)*n
    energy = omega(n,L,m)
    if normed:
        return np.array([[sqrt(energy-k)/sqrt(2*energy)],
                         [-sqrt(energy+k)/sqrt(2*energy)]])
    return np.array([[sqrt(energy-k)],[-sqrt(energy+k)]])

class FermionOperator():
    """ abstract class for normal ordered fermionic operator
    A normal-ordered operator is given by two lists of mode indices specifying
    which creation operators are present (clist) and which annihilation operators
    are present (dlist) and an overall multiplicative coefficient.
    
    For fermionic operators, we also have anticlist and antidlist for the
    antiparticles. To keep the sign conventions straight, we apply particle
    operators first and then antiparticle operators, but each of these is
    separately normal-ordered.
    
    Note also that the order matters in clist and dlist due to the signs
    associated with anticommutation. Operators are provided in the order one
    would write them (from left to right) and stored in reverse order
    (the order they act on states).
    
    Attributes:
        clist (list of ints): a list of ns (Fourier indices) corresponding 
        to creation ops
        dlist (list of ints): another list of ns corresponding to 
        destruction ops
        anticlist (list of ints)
        antidlist (list of ints)
        L (float): circumference of the circle (the spatial dimension)
        m (float): mass of the field
        coeff (float): the overall multiplicative prefactor of this operator
        deltaE (float): the net energy difference produced by acting on a state
            with this operator
    """
    def __init__(self,clist,dlist,anticlist,antidlist,
                 L,m,extracoeff=1,normed=False):
        """
        Args:
            clist, dlist, L, m: as above
            extracoeff (float): an overall multiplicative prefactor for the
                operator, *written as a power of the field operator phi*
            normed (bool): indicates whether factor of 1/sqrt(2*omega) has
                been absorbed into the definition of the spinor wavefunctions
        """
        # Check if there are multiple operators acting on the same mode.
        # Since fermionic operators anticommute, operators which have e.g.
        # 2 annihilation operators acting on the same mode are just 0.
        self.uniqueOps = (self.checkValidList(clist) 
                          and self.checkValidList(dlist)
                          and self.checkValidList(anticlist)
                          and self.checkValidList(antidlist))
        
        self.clist = clist[::-1]
        self.dlist = dlist[::-1]
        self.anticlist = anticlist[::-1]
        self.antidlist = antidlist[::-1]
        self.L=L
        self.m=m
        # coeff converts the overall prefactor of phi (extracoeff) to a prefactor
        # for the string of creation and annihilation operators in the final operator
        # see the normalization in Eq. 2.6
        if normed:
            self.coeff = extracoeff
        else:
            #note: have to be careful with this for massless zero modes
            self.coeff = extracoeff/product([sqrt(2.*L*omega(n,L,m))
                                             for n in clist+dlist])

        self.deltaE = sum([omega(n,L,m) for n in clist]) - sum([omega(n,L,m) for n in dlist])
        #can perform this as a vector op but the speedup is small
        #self.deltaE = sum(omega(array(clist),L,m))-sum(omega(array(dlist),L,m))
    
    def checkValidList(self, opsList):
        """
        Given a list of creation or annihilation operators, checks if
        the modes they act on are all unique.
        """
        
        return len(np.unique(opsList)) == len(opsList)
    
    def __repr__(self):
        return (str(self.clist) + " " + str(self.dlist) + " " + str(self.anticlist)
                + " " + str(self.antidlist) + " " + str(self.coeff))
    
    def _transformState(self, state0, returnCoeff=False, dressed=False):
        """
        Applies the normal ordered operator to a given state.
        
        Args:
            state0 (State): an input FermionState for this operator
            returncoeff (bool): boolean representing whether or not to
                include the factor self.coeff with the returned state
        
        Returns:
            A tuple representing the input state after being acted on by
            the normal-ordered operator and any multiplicative factors
            from performing the commutations.
        
        Example:
            For a state with nmax=1 and state0.occs = [0,0,2] 
            (2 excitations in the n=+1 mode, since counting starts at
            -nmax) if the operator is a_{k=1}, corresponding to
            clist = []
            dlist = [1]
            coeff = 1
            then this will return a state with occs [0,0,1] and a prefactor of
            2 (for the two commutations).
            
        """
        if not self.uniqueOps:
            return (0,None)
        #make a copy of this state up to occupation numbers and nmax
        #use DressedFermionState if the original state is dressed
        #otherwise use FermionState
        if state0.isDressed:
            state = DressedFermionState(particleOccs=state0.occs[:,0],
                             antiparticleOccs=state0.occs[:,1],
                             zeromode=state0.getAZeroMode(),
                             nmax=state0.nmax,
                             fast=True)
        else:
            state = FermionState(particleOccs=state0.occs[:,0],
                                 antiparticleOccs=state0.occs[:,1],
                                 nmax=state0.nmax,
                                 fast=True)
        
        # note: there may be an easier way for fermionic states
        # however, these loops are short and fast, so NumPy shortcuts probably
        # will not provide too much speed-up at this level.
        
        norm = 1
        
        for i in self.dlist:
            if state[i][0] == 0:
                return(0,None)
            state[i][0] -= 1
            # we have to anticommute past all the antiparticle creation ops
            # and the particle creation ops up to i
            norm *= (-1)**(np.sum(state.occs[:,1])+
                           np.sum(state.occs[:int(i-state.nmin),0]))
        
        for i in self.antidlist:
            if state[i][1] == 0:
                return(0,None)
            state[i][1] -= 1
            # anticommute past the antiparticle creation ops up to i
            norm *= (-1)**(np.sum(state.occs[:int(i-state.nmin),1]))
        
        for i in self.clist:
            # by Pauli exclusion, states can have at most one excitation
            # in a mode
            if state[i][0] == 1:
                return (0,None)
            state[i][0] += 1
            # anticommute past all the antiparticle creation ops and the
            # particle creation ops through i
            norm *= (-1)**(np.sum(state.occs[:,1])+
                           np.sum(state.occs[:int(i-state.nmin),0]))
        
        
        for i in self.anticlist:
            if state[i][1] == 1:
                return (0,None)
            state[i][1] += 1
            # anticommute past the antiparticle creation ops
            norm *= (-1)**(np.sum(state.occs[:int(i-state.nmin),1]))
        
        # We never pick up a nontrivial normalization factor for fermionic 
        # states since the occupation numbers are either one or zero.
        # The only option is if we wish to return the overall coefficient
        # of this operator or not.
        if returnCoeff:
            return (norm*self.coeff, state)
        return (norm, state)

    def apply(self, basis, i, lookupbasis=None):
        """ Takes a state index in basis, returns another state index (if it
        belongs to the lookupbasis) and a proportionality coefficient. Otherwise raises NotInBasis.
        lookupbasis can be different from basis, but it's assumed that they have the same nmax"""
        if lookupbasis == None:
            lookupbasis = basis
        if self.deltaE+basis[i].energy < 0.-tol or self.deltaE+basis[i].energy > lookupbasis.Emax+tol:
            # The transformed element surely does not belong to the basis if E>Emax or E<0
            raise NotInBasis()
        # apply the normal-order operator to this basis state
        n, newstate = self._transformState(basis[i])
        #if the state was annihilated by the operator, return (0, None)
        if n==0:
            return (0, None)
        # otherwise, look up this state in the lookup basis
        norm, j = lookupbasis.lookup(newstate)
        c = 1.
        #if basis[i].isParityEigenstate():
        #    c = 1/sqrt(2.)
            # Required for state normalization
        # return the overall proportionality and the state index
        return (norm*c*sqrt(n)*self.coeff, j)
    
    def apply2(self, basis, state, lookupbasis=None):
        """
        Like apply but with a state as input rather than an index in the
        original basis.
        """
        # TO-DO: add the energy shortcut from the original apply
        # we need the energy of the state-- is it expensive to recompute it?
        if lookupbasis == None:
            lookupbasis = basis
        
        n, newstate = self._transformState(state)
        
        if n==0:
            return (0, None)
        
        norm, j = lookupbasis.lookup(newstate)
        
        c = 1.
        #no sqrt n since n is either 1 or -1
        return (norm*c*n*self.coeff,j)