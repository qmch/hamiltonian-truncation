######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import scipy
from scipy import pi
import numpy as np
import scipy.sparse.linalg
import scipy.sparse
import scipy.interpolate
from operator import attrgetter
from math import factorial
from statefuncs import Basis, NotInBasis, omega, State
from oscillators import NormalOrderedOperator as NOO
import collections
import renorm
import itertools

import time

tol = 0.0001

""" P denotes spatial parity, while K field parity. For now only the P-even sector is implemented """

def comb(*x):
    """ computes combinatorial factor for list of elements """
    #note: we could do this with np.unique if the list was a numpy array
    #print(collections.Counter(x).values())
    return factorial(len(x))/np.prod(scipy.special.factorial(list(collections.Counter(x).values())))
    #return factorial(len(x))/np.prod(list(map(factorial,collections.Counter(x).values())))
    #return factorial(len(x))/scipy.prod(set(map(factorial,collections.Counter(x).values())))

class Matrix():
    """ Matrix with specified state bases for row and column indexes. 
    This class is useful to easily extract submatrices """
    def __init__(self, basisI, basisJ, M=None):
        self.basisI = basisI
        self.basisJ = basisJ
        
        #if no M is given, set M to be a basisI.size by 1 sparse matrix.
        if(M == None):
            self.M = scipy.sparse.coo_matrix((basisI.size, 1))
        else:
            self.M = M
            self.check()
        
    def addColumn(self, newcolumn):
        m = scipy.sparse.coo_matrix(newcolumn).transpose()
        self.M = scipy.sparse.hstack([self.M,m])
    
    def finalize(self):
        """Drops first column of M and ensures M is in COO format"""
        self.M = self.M.tocsc()[:,1:].tocoo()
        self.check()
    
    def check(self):
        """Check that M has the right dimensions"""
        if self.M.shape != (self.basisI.size, self.basisJ.size):
            raise ValueError('Matrix shape inconsistent with given bases')

    def __add__(self, other):
        """ Sum of matrices """

        return Matrix(self.basisI, self.basisJ, self.M+other.M)
    
    def __mul__(self, other):
        """ Multiplication of matrix with matrix or number"""
        if(other.__class__ == self.__class__):
            return Matrix(self.basisI, other.basisJ, self.M*other.M)
        else:
            return Matrix(self.basisI, self.basisJ, self.M*float(other))

    def to(self, form):
        """ Format conversion """
        return Matrix(self.basisI, self.basisJ, self.M.asformat(form))
    
    def sub(self, subBasisI=None, subBasisJ=None):
        """ This extracts a submatrix given a subspace of the initial vector space, both for rows and columns """
    
        if subBasisI != None and subBasisJ != None:
            rows = [self.basisI.lookup(state)[1]  for state in subBasisI]
            columns = [self.basisJ.lookup(state)[1]  for state in subBasisJ]
            return Matrix(subBasisI, subBasisJ, self.M.tocsr()[scipy.array(rows)[:,scipy.newaxis],scipy.array(columns)])
        
        elif subBasisI != None and subBasisJ == None:
            rows = [self.basisI.lookup(state)[1]  for state in subBasisI]        
            return Matrix(subBasisI, self.basisJ, self.M.tocsr()[scipy.array(rows),:])

        elif subBasisI == None and subBasisJ != None:
            columns = [self.basisJ.lookup(state)[1]  for state in subBasisJ]        
            return Matrix(self.basisI, subBasisJ, self.M.tocsr()[:,scipy.array(columns)])

        else:
            return self
    
    def transpose(self):
        
        """Transpose the matrix (switch basisI and basisJ and transpose M)"""
        return Matrix(self.basisJ, self.basisI, self.M.transpose())
    
    def __repr__(self):
        return str(self.M)

class Phi1234():
    """ main class 
    
    Attributes
    ----------
    L : float
        Circumference of the circle on which to quantize
    m : float
        Mass of the scalar field
    Emax: float
        Maximum energy cutoff
    
    All basic dictionaries in this class have two keys, 1 and -1, 
        corresponding to values of k-parity.
    
    h0: dict of Matrix objects
        Values are Matrix objects corresponding to the free Hamiltonian
    potential: dict of dict of Matrix
        Values are dictionaries where the keys are orders in
        the field (0,2,4 correspond to phi^0,phi^2,phi^4) and the values
        are the potential matrices at each order stored as Matrix objects
        
    When actually diagonalizing the Hamiltonian, we further restrict
        to a subspace with some different nmax, possibly?
    
    H: dict of Matrix objects
        Like h0, but on a restricted subspace defined by basis rather than
        fullBasis
    V: dict of dict of Matrix
        Like potential but restricted to the basis subspace
    
    """
    def __init__(self):
        self.L = None
        self.m = None
        self.Emax = None
        
        self.h0 = {1: None, -1: None}
        self.potential = {1 :{ }, -1:{ }}
        self.h0Sub = {1: None, -1: None}
        self.H = {1: None, -1: None}
        self.V = {1: {}, -1: {}}

        self.eigenvalues = {1: None, -1: None}
        self.eigsrenlocal = {1: None, -1: None}
        self.eigsrensubl = {1: None, -1: None}
        self.eigenvectors = {1: None, -1: None}
        # Eigenvalues and eigenvectors for different K-parities

        self.basis = {1: None, -1: None}
        self.fullBasis = {1: None, -1: None}

    def buildFullBasis(self,k,L,m,Emax):
        """ Builds the full Hilbert space basis """

        self.L=float(L)
        self.m=float(m)
        
        #create a Basis object (see statefuncs.py)
        #and set it as self.fullBasis
        #a Basis contains a list of states, each state defined by a set of 
        #occupation numbers for each Fourier mode
        self.fullBasis[k] = Basis(L=self.L, Emax=Emax, m=self.m, K=k)


    def buildBasis(self,k,Emax):
        """
        Builds the Hilbert space basis for which the Hamiltonian to actually diagonalize
        is calculated (in general it's a subspace of fullBasis) 
        
        Note that this is called in phi4eigs, but not in generating
        the original potential matrix and free Hamiltonian.
        """

        self.basis[k] = Basis(m=self.m, L=self.L, Emax=Emax, K=k, nmax=self.fullBasis[k].nmax)
        # We use the vector length (nmax) of the full basis. In this way we can compare elements between the two bases
        self.Emax = float(Emax)

        for nn in (0,2,4):
            self.V[k][nn] = self.potential[k][nn].sub(self.basis[k], self.basis[k]).M.tocoo()

        self.h0Sub[k] = self.h0[k].sub(self.basis[k],self.basis[k]).M.tocoo()

    def buildMatrix(self):
        """ Builds the full Hamiltonian in the basis of the free Hamiltonian eigenvectors.
        This is computationally intensive. It can be skipped by loading the matrix from file """
        
        """
        Possible speedups:
            Simplify the diagonal operator loops?
            Can we apply a numpy mask to simplify the loops without the
            conditionals?
        
        Basically this loops over the operators at each order in phi
        and stores all the normal order operators explicitly in lists.
        """
        L=self.L
        m=self.m

        for k in (1,-1):
            basis = self.fullBasis[k]
            lookupBasis = self.fullBasis[k]
            Emax = basis.Emax
            nmax = basis.nmax

            diagOps = {0: None, 2:None, 4:None}
            offdiagOps = {0: None, 2:None, 4:None}

            diagOps[0] = [ NOO([],[],L,m) ]
            
            offdiagOps[0] = []
            #the 2 is a combinatorial factor since both aa^dagger and a^dagger a contribute
            diagOps[2] = [ NOO([a],[a],L,m, extracoeff=2.) for a in range(-nmax,nmax+1) ]
            #the symmetry factor is 1 if a=-a and 2 otherwise
            offdiagOps[2] = [ NOO([a,-a],[],L,m,extracoeff=comb(a,-a))
                    for a in range(-nmax,nmax+1) if a<=-a<=nmax and
                    omega(a,L,m)+omega(-a,L,m) <= Emax+tol]
            # the default symmetry factor is 6 (4 choose 2) if a and b are distinct
            # and c, a+b-c are distinct
            # notice the index for b runs from a to nmax so we only get unique
            # pairs a and b, i.e. (1,1), (1,2), (1,3), (2,2), (2,3), (3,3).
            diagOps[4] = [ NOO([a,b],[c,a+b-c],L,m, extracoeff=6.*comb(a,b)*comb(c,a+b-c))
                    for a in range(-nmax,nmax+1) for b in range (a,nmax+1)
                    for c in range(-nmax,nmax+1) if
                    ( c<=a+b-c<=nmax
                    and (a,b) == (c,a+b-c) 
                    and -Emax-tol <= omega(a,L,m)+omega(b,L,m) - omega(c,L,m)-omega(a+b-c,L,m) <=Emax+tol)]
                
            offdiagOps[4] = [ NOO([a,b,c,-a-b-c],[],L,m,extracoeff=comb(a,b,c,-a-b-c))
                    for a in range(-nmax,nmax+1) for b in range (a,nmax+1)
                    for c in range(b,nmax+1) if c<=-a-b-c<=nmax and
                    omega(a,L,m)+omega(b,L,m) + omega(c,L,m)+omega(-a-b-c,L,m)<= Emax+tol]  \
                + [ NOO([a,b,c],[a+b+c],L,m, extracoeff = 4. * comb(a,b,c))
                    for a in range(-nmax, nmax+1) for b in range (a,nmax+1)
                    for c in range(b,nmax+1) if
                    (-nmax<=a+b+c<=nmax
                    and -Emax-tol <= omega(a,L,m)+omega(b,L,m)+ omega(c,L,m)-omega(a+b+c,L,m) <=Emax+tol)] \
                + [ NOO([a,b],[c,a+b-c],L,m, extracoeff = 6. * comb(a,b)*comb(c,a+b-c))
                    for a in range(-nmax,nmax+1) for b in range (a,nmax+1)
                    for c in range(-nmax,nmax+1) if
                    ( c<=a+b-c<=nmax
                    and (a,b) != (c,a+b-c)
                    and sorted([abs(a),abs(b)]) < sorted([abs(c),abs(a+b-c)])
                    and -Emax-tol <= omega(a,L,m)+omega(b,L,m)- omega(c,L,m)-omega(a+b-c,L,m) <=Emax+tol)]

            #save as h0 a lookupBasis.size by 1 sparse matrix initialized to zeros
            #really just a single column
            #and store the bases as h0[k].basisI and h0[k].basisJ
            #self.h0[k] = Matrix(lookupBasis, basis)
            
            tempEnergies = np.empty(basis.size)
            #initialTime = time.time()
            #this was xrange in the original; range in 3.x was xrange in 2.x -IL
            for j in range(basis.size):
                #make a new column of the appropriate length
                #newcolumn = scipy.zeros(lookupBasis.size)
                #set the jth entry in this column to be the energy of
                #the jth state in the basis
                #newcolumn[j] = basis[j].energy
                tempEnergies[j] = basis[j].energy
                #self.h0[k].addColumn(newcolumn)
                
            #self.h0[k].finalize()            
            """
                basically this is just a diagonal matrix of the eigenvalues
                since the Hamiltonian h0 is diagonal in this basis this is faster.
                the loop adding columns takes 0.45711565017700195 s
                just creating the sparse matrix takes 0.000997304916381836 s.
            """
            temph0 = scipy.sparse.diags(tempEnergies,format="coo")
            self.h0[k] = Matrix(lookupBasis,basis,temph0)
            
            #ran some tests, doing scipy.sparse.diags does seem more straightforward
            #print(time.time()-initialTime)
            
            """
                We build the potential in this basis. self.potential is a
                dictionary with two keys corresponding to k=1 and k=-1.
                Each of the entries is a list of sparse matrices.
                Each sparse matrix consists of a sum of off-diagonal components
                and diagonal components in COO format.
                
                Possible speedup: could we apply each operator to all states
                in one shot? Then we would just loop over operators.
                
                Right now, for each state, we apply each operator one at a time
                and mark down its image in the corresponding column (the origin
                state) and row (the image state).
                
                If we apply an operator to a vector of states, we will get
                a vector of their images which can be collectively checked for
                validity by checking the dictionary keys/energy ranges, and
                then this is just the matrix form of the operator.
            """
            # for each order (0,2,4) in phi
            for n in offdiagOps.keys():

                offdiag_V = Matrix(lookupBasis, basis)
                diagonal = np.zeros(basis.size)
                    
                # for each state in the basis
                for j in range(basis.size):
                                        
                    newcolumn = np.zeros(lookupBasis.size)
                    # for each off-diagonal operator at a given order
                    for op in offdiagOps[n]:
                        try:
                            # apply this operator to find whether the
                            # new state is still in the basis
                            (x,i) = op.apply(basis,j,lookupBasis)
                            # if so, add the corresponding value to the matrix
                            # this is basically writing the effects of the
                            # operator in the basis of the free states
                            if(i != None):
                                newcolumn[i]+=x
                        except NotInBasis:
                            pass

                    offdiag_V.addColumn(newcolumn)
                    
                    # for each diagonal operator at the same order n
                    for op in diagOps[n]:
                        
                        (x,i) = op.apply(basis,j,lookupBasis)
                        # It should be j=i
                        
                        if i!= None:
                            if i != j:
                                raise RuntimeError('Non-diagonal operator')                            
                            diagonal[i]+=x

                offdiag_V.finalize()
                diag_V = scipy.sparse.spdiags(diagonal,0,basis.size,basis.size)
                
                self.potential[k][n] = (offdiag_V+offdiag_V.transpose()+Matrix(lookupBasis, basis, diag_V)).to('coo')*self.L


    def saveMatrix(self, fname):
        """ Saves the free Hamiltonian and potential matrices to file """

        t = (fname, self.L, self.m, \
            self.fullBasis[1].Emax, self.fullBasis[1].nmax, \
            self.fullBasis[-1].Emax, self.fullBasis[-1].nmax, \
            self.h0[1].M.data,self.h0[1].M.row,self.h0[1].M.col, \
            self.potential[1][0].M.data,self.potential[1][0].M.row,self.potential[1][0].M.col, \
            self.potential[1][2].M.data,self.potential[1][2].M.row,self.potential[1][2].M.col, \
            self.potential[1][4].M.data,self.potential[1][4].M.row,self.potential[1][4].M.col, \
            self.h0[-1].M.data,self.h0[-1].M.row,self.h0[-1].M.col, \
            self.potential[-1][0].M.data,self.potential[-1][0].M.row,self.potential[-1][0].M.col, \
            self.potential[-1][2].M.data,self.potential[-1][2].M.row,self.potential[-1][2].M.col, \
            self.potential[-1][4].M.data,self.potential[-1][4].M.row,self.potential[-1][4].M.col \
            )
        scipy.savez(*t)

    def loadMatrix(self, fname):
        """ Loads the free Hamiltonian and potential matrices from file """

        f = scipy.load(fname)
        self.L = f['arr_0'].item()
        self.m = f['arr_1'].item()

        Emax = {1:f['arr_2'].item(), -1:f['arr_4'].item()}
        nmax = {1:f['arr_3'].item(), -1:f['arr_5'].item()}
                
        for i, k in enumerate((1,-1)):
            n = 12
            z = 6
                
            self.buildFullBasis(L=self.L, m=self.m, Emax=Emax[k], k=k)

            basisI = self.fullBasis[k]
            basisJ = self.fullBasis[k]

            self.h0[k] = Matrix(basisI, basisJ, scipy.sparse.coo_matrix((f['arr_'+(str(z+i*n))], (f['arr_'+(str(z+1+i*n))], f['arr_'+(str(z+2+i*n))])), shape=(basisI.size, basisJ.size)))
            self.potential[k][0] = Matrix(basisI, basisJ, scipy.sparse.coo_matrix((f['arr_'+(str(z+3+i*n))], (f['arr_'+(str(z+4+i*n))], f['arr_'+(str(z+5+i*n))])), shape=(basisI.size, basisJ.size)))
            self.potential[k][2] = Matrix(basisI, basisJ, scipy.sparse.coo_matrix((f['arr_'+(str(z+6+i*n))], (f['arr_'+(str(z+7+i*n))], f['arr_'+(str(z+8+i*n))])), shape=(basisI.size, basisJ.size)))
            self.potential[k][4] = Matrix(basisI, basisJ, scipy.sparse.coo_matrix((f['arr_'+(str(z+9+i*n))], (f['arr_'+(str(z+10+i*n))], f['arr_'+(str(z+11+i*n))])), shape=(basisI.size, basisJ.size)))
 
    def setcouplings(self, g4, g2=0.):
        self.g2 = float(g2)
        self.g4 = float(g4)
    
    def renlocal(self,Er):
        self.g0r, self.g2r, self.g4r = renorm.renlocal(self.g2,self.g4,self.Emax,Er)
        self.Er = Er    

    def computeHamiltonian(self, k=1, ren=False):
        """ Computes the (renormalized) Hamiltonian to diagonalize
        k : K-parity quantum number
        ren : if True, computes the eigenvalue with the "local" renormalization procedure, otherwise the "raw" eigenvalues 
        """
        if not(ren):
            self.H[k] = self.h0Sub[k] + self.V[k][2]*self.g2 + self.V[k][4]*self.g4
        else:
            self.H[k] = self.h0Sub[k] + self.V[k][0]*self.g0r + self.V[k][2]*self.g2r + self.V[k][4]*self.g4r
    

    def computeEigval(self, k=1, ren=False, corr=False, sigma=0, n=10):
        """ Diagonalizes the Hamiltonian and possibly computes the subleading renormalization corrections
        k (int): K-parity quantum number 
        ren (bool): it should have the same value as the one passed to computeHamiltonian()
        corr (bool): if True, computes the subleading renormalization corrections, otherwise not.
        n (int): number of lowest eigenvalues to compute
        sigma : value around which the Lanczos method looks for the lowest eigenvalue. 
        """

        if not ren:
            (self.eigenvalues[k], eigenvectorstranspose) = scipy.sparse.linalg.eigsh(self.H[k], k=n, sigma=sigma,
                            which='LM', return_eigenvectors=True)
        else:
            (self.eigsrenlocal[k], eigenvectorstranspose) = scipy.sparse.linalg.eigsh(self.H[k], k=n, sigma=sigma,
                            which='LM', return_eigenvectors=True)
        eigenvectors = eigenvectorstranspose.T
        
        if corr:
            print("Adding subleading corrections to k="+str(k), " eigenvalues")

            self.eigsrensubl[k] = np.zeros(n)
            cutoff = 5.

            for i in range(n):
                cbar = eigenvectors[i]
                if abs(sum([x*x for x in cbar])-1.0) > 10**(-13):
                    raise RuntimeError('Eigenvector not normalized')

                Ebar = self.eigsrenlocal[k][i]
                self.eigsrensubl[k][i] += Ebar
                ktab, rentab = renorm.rensubl(self.g2, self.g4, Ebar, self.Emax, self.Er, cutoff=cutoff)

                tckren = { }
                tckren[0] = scipy.interpolate.interp1d(ktab,rentab.T[0],kind='linear')
                tckren[2] = scipy.interpolate.interp1d(ktab,rentab.T[1],kind='linear')
                tckren[4] = scipy.interpolate.interp1d(ktab,rentab.T[2],kind='linear')

                for nn in (0,2,4):
                    #for a,b,Vab in itertools.izip(self.V[k][nn].row,self.V[k][nn].col,self.V[k][nn].data):
                    for a,b,Vab in zip(self.V[k][nn].row,self.V[k][nn].col,self.V[k][nn].data):
                        if a > b:
                            continue
                        elif a == b:
                            c = 1
                        else:
                            c = 2

                        Eab2= (self.basis[k][a].energy + self.basis[k][b].energy)/2.
                        coeff = tckren[nn](Eab2)
                        self.eigsrensubl[k][i] += c * coeff * cbar[a] * cbar[b] * Vab

    def vacuumE(self, ren="raw"):
        if ren=="raw":
            return self.eigenvalues[1][0]
        elif ren=="renlocal":    
            return self.eigsrenlocal[1][0]
        elif ren=="rensubl":
            return self.eigsrensubl[1][0]
        else:
            raise ValueError("Wrong argument")
        # The vacuum is K-even

    def spectrum(self, k, ren="raw"):
        if ren=="raw":
            eigs = self.eigenvalues
        elif ren=="renlocal":    
            eigs = self.eigsrenlocal
        elif ren=="rensubl":
            eigs = self.eigsrensubl
        else:
            raise ValueError("Wrong argument")
        
        # Now subtract vacuum energies
        if k==1:
            return scipy.array([x-self.vacuumE(ren=ren) for x in eigs[k][1:]])
        elif k==-1:
            return scipy.array([x-self.vacuumE(ren=ren) for x in eigs[k]])
        else:
            raise ValueError("Wrong argument")
