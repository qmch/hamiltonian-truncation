# hamiltonian-truncation

Code for performing the numerical analysis of quantum field theories via the
Hamiltonian truncation method, based on work by Slava Rychkov and Lorenzo
Vitale.

Create and save to file the potential matrices, for L=6 and Emax=22:

$ python genMatrix.py 6 22

Calculate the spectrum for g=1, L=6 and Emax=20:

$ python phi4eigs.py Emax=22.0_L=6.0.npz 1 20

Tests can be run using the scripts in the tests folder, which are
otherwise self-contained:

$ python ./tests/testBasis.py