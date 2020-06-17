# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:43:17 2020

@author: aura1
"""

import sys
import numpy
from statefuncs import Basis, NotInBasis, omega, State
from oscillators import NormalOrderedOperator as NOO
import time

class npState(State):
    def __init__(self, occs, nmax, L=None, m=None, fast=False, checkAtRest=True):
        State.__init__(self, occs, nmax, L, m, fast, checkAtRest)
        self.occs = numpy.array(occs)
        
        if self.size == 2*self.nmax+1 and numpy.array_equal(self.occs[::-1],self.occs):
            self.__parityEigenstate = True
        else:
            self.__parityEigenstate = False

def main(argv):
    print("Hello world")
    #arguments to NOO are clist, dlist, L, m, and extracoeff (optional)
    L = 1
    m = 1
    myOperator = NOO([],[1],L,m)
    print(myOperator)
    nmax = 1
    myState = npState([2,0,2],nmax,L,m)
    print(myState)
    print(myState[numpy.array([0,0,-1])])
    
    occs = numpy.array([1,0,5])
    dlist = numpy.array([-1,1,1,1,1])
    uniqueCounts = numpy.unique(dlist,return_counts=True)
    dlist2 = numpy.vstack((uniqueCounts[0],uniqueCounts[1])).transpose()
    print(dlist2)
    start = time.time()
    for n in range(50000):
        npTest(occs,dlist2)
    print(time.time()-start)
    start = time.time()
    for n in range(50000):
        oldTest(occs,dlist)
    print(time.time()-start)

def npTest(myOccs,mydlist):
    #occs = myOccs
    #dlist = mydlist#numpy.array([[-1,1],[1,2]])
    nmax=1
    
    '''
    print("occs is ", occs)
    print("dlist is", dlist)
    print("dlist[:,0] is",dlist[:,0])
    print("occs[dlist[:,0]+nmax] is",occs[dlist[:,0]+nmax])
    print("dlist[:,1] is",dlist[:,1])
    print("occs[dlist[:,0]+nmax]-dlist[:,1] is",occs[dlist[:,0]+nmax]-dlist[:,1])
    '''
    myOccs[mydlist[:,0]+nmax] -= mydlist[:,1]
    #print("new version of occs is",occs)

def oldTest(myOccs,mydlist):
    #print("Called oldTest")
    occs = [1,0,5]#myOccs
    dlist = [-1,1,1,1,1]# mydlist#[-1,1,1]
    nmax=1
    n = 1.
    #for each of the destruction operators
    for i in dlist:
        #if there is no Fourier mode at that value of n (ground state)
        if occs[i+nmax] == 0:
            #then the state is annihilated
            return None
        #otherwise we multiply n by the occupation number of that state
        n *= occs[i+nmax]
        #and decrease its occupation number by 1
        occs[i+nmax] -= 1
    #print("old way says",occs)
    #return occs

if __name__ == "__main__":
    main(sys.argv)