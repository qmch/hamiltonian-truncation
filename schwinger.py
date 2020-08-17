######################################################
# 
# Fock space Hamiltonian truncation for QED in 1+1 dimensions
# Author: Ian Lim (itlim@ucdavis.edu), adapted from work by Ryzhkov and Vitale
# July 2020
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
from phi1234 import Matrix
from statefuncs import Basis, NotInBasis, omega, State
from oscillators import NormalOrderedOperator as NOO
from qedops import uspinor, vspinor, FermionOperator
from qedstatefuncs import FermionBasis, FermionState
import collections
import renorm
import itertools

import time

tol = 0.0001

""" P denotes spatial parity, while K field parity. 
For now only the P-even sector is implemented """

class Schwinger():
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
        
        self.h0 = None
        self.potential = None
        self.h0Sub = None
        self.H = None
        self.V = None

        self.eigenvalues = None
        self.eigsrenlocal = None
        self.eigsrensubl = None
        self.eigenvectors = None

        self.basis = None
        self.fullBasis = None

    def buildFullBasis(self,L,m,Emax):
        """ Builds the full Hilbert space basis """

        self.L=float(L)
        self.m=float(m)
        
        self.fullBasis = FermionBasis(self.L, Emax, self.m)

    # TO-DO: update this when computing eigenvals
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

        
        basis = self.fullBasis
        lookupBasis = self.fullBasis
        Emax = basis.Emax
        nmax = basis.nmax
        
        """
        We split operators into diagonal and off-diagonal operators.
        The idea is that for off-diagonal operators, we can just compute
        the image of half the operators and generate the conjugates by
        taking the transpose. That is, if an operator X takes state A to B,
        then X^\dagger should take B to A.
        """
        
        diagOps = {0: None, 2:None, 4:None}
        offdiagOps = {0: None, 2:None, 4:None}
        
        """
        diagOps[0] = [ NOO([],[],L,m) ]
        
        offdiagOps[0] = []

        diagOps[2] = [ NOO([a],[a],L,m, extracoeff=2.) for a in range(-nmax,nmax+1) ]

        offdiagOps[2] = [ NOO([a,-a],[],L,m,extracoeff=comb(a,-a))
                for a in range(-nmax,nmax+1) if a<=-a<=nmax and
                omega(a,L,m)+omega(-a,L,m) <= Emax+tol]
    
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
        """
        
        #save as h0 a lookupBasis.size by 1 sparse matrix initialized to zeros
        #really just a single column
        #and store the bases as h0[k].basisI and h0[k].basisJ
        #self.h0[k] = Matrix(lookupBasis, basis)
        
        tempEnergies = np.empty(basis.size)
        #tempEnergies = np.array([basis[j].energy
        #                         for j in range(basis.size)])
        for j in range(basis.size):
            tempEnergies[j] = basis[j].energy
            
        #self.h0[k].finalize()            
        temph0 = scipy.sparse.diags(tempEnergies,format="coo")
        self.h0 = Matrix(lookupBasis,basis,temph0)
        
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
        
        psidaggerpsi = self.generateOperators()
        
        # can change this to enumerate if we want to specialize to diagonal
        # operators
        
        # the use of only the upper triangular matrix and then just transposing
        # is a great way to save time
        
        # maybe later should just enumerate all 16 ops by hand? Seems like
        # a pain though.
        
        potential = Matrix(lookupBasis,basis)
        
        for state in basis:
            
            newcolumn = np.zeros(lookupBasis.size)
            
            for k in np.arange(-nmax,nmax+1):
                
                for op2 in psidaggerpsi[-k]:                    
                    n, newstate= op2._transformState(state,returnCoeff=True)
                    if n == 0:
                        continue
                    for op1 in psidaggerpsi[k]:
                        try:
                            normalization, index = op1.apply2(basis,newstate)
                        
                            if (index != None):
                                newcolumn[index] += normalization * n / (k**2)
                        except NotInBasis:
                            pass
            
            potential.addColumn(newcolumn)
        
        potential.finalize()
        print(potential.M.toarray())
        isSymmetric = (np.array_equal(potential.M.toarray(),
                                      potential.M.toarray().T))
        if isSymmetric:
            print("Matrix is symmetric")
        
        self.potential = potential
        # for each order (0,2,4) in phi
        """
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
            
            self.potential[n] = (offdiag_V+offdiag_V.transpose()+Matrix(lookupBasis, basis, diag_V)).to('coo')*self.L
        """

    # TO-DO: update for QED
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
 
    def generateOperators(self):
        opsList = {}
        
        # we will generate (psi^\dagger psi)_k
        for k in np.arange(-self.fullBasis.nmax,self.fullBasis.nmax+1):
            opsList[k] = []
            if k == 0:
                continue
            # note: what is the range on n? revisit. All valid ns such that
            # -nmax <= k+n <= nmax and -nmax <= n <= nmax
            for n in np.arange(-self.fullBasis.nmax,self.fullBasis.nmax+1):
                if not (-self.fullBasis.nmax <= k-n <= self.fullBasis.nmax):
                    continue
                if k-n == 0 or n == 0:
                    continue
                
                coeff = np.vdot(uspinor(n-k,self.L,self.m,normed=True),
                                uspinor(n,self.L,self.m,normed=True))
                adaggera = FermionOperator([n-k],[n],[],[],self.L,self.m,
                                           extracoeff=coeff,normed=True)
                #n.b. -1 from anticommutation of bdagger and adagger
                coeff = -1 * np.vdot(uspinor(n-k,self.L,self.m,normed=True),
                                vspinor(-n,self.L,self.m,normed=True))
                bdaggeradagger = FermionOperator([n-k],[],[-n],[],self.L,self.m,
                                                 extracoeff=coeff,normed=True)
                
                coeff = np.vdot(vspinor(k-n,self.L,self.m,normed=True),
                                uspinor(n,self.L,self.m,normed=True))
                ba = FermionOperator([],[k-n],[],[n],self.L,self.m,
                                     extracoeff=coeff,normed=True)
                # -1 from anticommuting b and b dagger
                coeff = -1 * np.vdot(vspinor(k-n,self.L,self.m,normed=True),
                                     vspinor(-n,self.L,self.m,normed=True))
                bdaggerb = FermionOperator([],[],[-n],[k-n],self.L,self.m,
                                           extracoeff=coeff,normed=True)
                #the anticommutator is always trivial because k-n = -n -> k=0
                #and we've precluded this possiblity since k != 0
                
                opsList[k] += [adaggera, bdaggeradagger, ba, bdaggerb]
                
        return opsList

    def generateOperators2(self,nmax):
        #an attempt at writing down all the operators explicitly
        #...let's not do this maybe
        
        op0000 = [FermionOperator([k1,k3],[k2,k1+k3-k2],[],[],self.L,self.m,
                                  extracoeff=-1/(k1+k2)^2,normed=True) 
                  for k1 in range(-nmax,nmax+1) for k2 in range(-nmax,nmax+1)
                  for k3 in range(-nmax,nmax+1)
                  if (k1+k2 != 0) and (k1 != 0) and (k2 != 0) and (k3 != 0)
                  ]
        
        #needs spinor wavefunction inner products
        op51a = [FermionOperator([k1,k3],[k2,-k1-k2-k3],[],[],self.L,self.m,
                                 extracoeff=-1/(k1+k2)**2,normed=True)
                 for k1 in range(-nmax,nmax+1) for k2 in range(-nmax,nmax+1)
                 for k3 in range(-nmax,nmax+1)
                 if (k1+k2 != 0) and (k1 != 0) and (k2 != 0) and (k3 != 0)
                 ]
        
        op51a_2 = [FermionOperator([k1],[-k1-k2-k3],[],[],self.L,self.m,
                                   extracoeff=-1/(k1+k2)**2,normed=True)
                   for k1 in range(-nmax,nmax+1) for k2 in range(-nmax,nmax+1)
                   for k3 in range(-nmax,nmax+1)
                   if (k2==k3) and (k1+k2 != 0) and (k1 != 0) and (k2 != 0)
                   #k3 nonzero guaranteed since k2==k3
                   ]

    def setcoupling(self, g):
        self.g = float(g)
    
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
