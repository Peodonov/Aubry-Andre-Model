import numpy as np
from scipy import fft
from scipy import integrate
from scipy.sparse import csc_matrix
from quspin.operators import hamiltonian, exp_op #Hamiltonians
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis
import matplotlib.pyplot as plt

L = 20 # Length of lattice
J = 1  # hopping strength
beta = (1 + np.sqrt(5))/2 # periodicity
phi = 0 # phase
lam = 2*J # disorder strength
U = 0
pbc = False # boundary condition (true = Periodic PBC, false = Open OBC)
mu = lam*np.cos(2*np.pi*beta*np.arange(L) + phi)


basis = spinless_fermion_basis_1d(L = L, Nf = 1, a = 1)
print(basis)


n_pot=[[-mu[i],i] for i in range(L)]
J_nn_right=[[-J,i,(i+1)%L] for i in range(L)] # PBC
J_nn_left=[[+J,i,(i+1)%L] for i in range(L)] # PBC
U_nn_int=[[U,i,(i+1)%L] for i in range(L)] # PBC

no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

static = [['+-', J_nn_left], ['-+', J_nn_right],['n', n_pot]]

Ham = hamiltonian(static, [], dtype = np.float64, basis = basis, **no_checks)




