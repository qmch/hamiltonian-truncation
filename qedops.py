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

tol = 0.0001

def uspinor(n,L,m):
    k = (2.*pi/L)*n
    energy = omega(n,L,m)
    return np.array([[sqrt(energy+k)],[sqrt(energy-k)]])

def vspinor(n,L,m):
    k = (2.*pi/L)*n
    energy = omega(n,L,m)
    return np.array([[sqrt(energy+k)],[-sqrt(energy-k)]])

class FermionOperator():
    """ abstract class for normal ordered fermionic operator
    A normal-ordered operator is given by two lists of (Fourier) indices specifying
    which creation operators are present (clist) and which annihilation operators
    are present (dlist) and an overall multiplicative coefficient.
    
    Attributes:
        clist (list of ints): a list of ns (Fourier indices) corresponding 
        to creation ops, see phi1234.py buildMatrix()
        dlist (list of ints): another list of ns corresponding to 
        destruction ops
        L (float): circumference of the circle (the spatial dimension)
        m (float): mass of the field
        coeff (float): the overall multiplicative prefactor of this operator
        deltaE (float): the net energy difference produced by acting on a state
            with this operator
    """
    def __init__(self,leftclist,leftdlist,rightclist,rightdlist,
                 L,m,extracoeff=1):
        """
        Args:
            clist, dlist, L, m: as above
            extracoeff (float): an overall multiplicative prefactor for the
                operator, *written as a power of the field operator phi*
        """
        self.clist=clist
        self.dlist=dlist
        self.L=L
        self.m=m
        # coeff converts the overall prefactor of phi (extracoeff) to a prefactor
        # for the string of creation and annihilation operators in the final operator
        # see the normalization in Eq. 2.6
        self.coeff = extracoeff/product([sqrt(2.*L*omega(n,L,m)) for n in clist+dlist])
        #can this be sped up by vectorizing the omega function? IL
        self.deltaE = sum([omega(n,L,m) for n in clist]) - sum([omega(n,L,m) for n in dlist])
        #can perform this as a vector op but the speedup is small
        #self.deltaE = sum(omega(array(clist),L,m))-sum(omega(array(dlist),L,m))
        
    def __repr__(self):
        return str(self.clist)+" "+str(self.dlist) 
    
    def _transformState(self, state0):        
        """
        Applies the normal ordered operator to a given state.
        
        Args:
            state0 (State): an input state for this operator
        
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
        #make a copy of this state up to occupation numbers and nmax
        state = State(state0.occs[:], state0.nmax, fast=True)
        n = 1.
        #for each of the destruction operators
        for i in self.dlist:
            #if there is no Fourier mode at that value of n (ground state)
            if state[i] == 0:
                #then the state is annihilated
                return(0,None)
            #otherwise we multiply n by the occupation number of that state
            n *= state[i]
            #and decrease its occupation number by 1
            state[i] -= 1
        #for each of the creation operators
        for i in self.clist:
            #multiply n by the occupation number of that state
            n *= state[i]+1
            #increase the occupation number of that mode by 1
            state[i] += 1
        return (n, state)

    def apply(self, basis, i, lookupbasis=None):
        """ Takes a state index in basis, returns another state index (if it
        belongs to the lookupbasis) and a proportionality coefficient. Otherwise raises NotInBasis.
        lookupbasis can be different from basis, but it's assumed that they have the same nmax"""
        if lookupbasis == None:
            lookupbasis = basis
        if self.deltaE+basis[i].energy < 0.-tol or self.deltaE+basis[i].energy > lookupbasis.Emax+tol:
            # The trasformed element surely does not belong to the basis if E>Emax or E<0
            raise NotInBasis()
        # apply the normal-order operator to this basis state
        n, newstate = self._transformState(basis[i])
        #if the state was annihilated by the operator, return (0, None)
        if n==0:
            return (0, None)
        # otherwise, look up this state in the lookup basis
        m, j = lookupbasis.lookup(newstate)
        c = 1.
        if basis[i].isParityEigenstate():
            c = 1/sqrt(2.)
            # Required for state normalization
        # return the overall proportionality and the state index
        return (m*c*sqrt(n)*self.coeff, j)
