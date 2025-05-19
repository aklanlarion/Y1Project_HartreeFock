'''
This file containts the code for running the SCF algorithm. The inputs are given matrices & 
functions performing the operations, and its output is a .cube file. 
'''

#importing necessary packages
import numpy as np
from scipy import linalg
import matrices as mat

#---Define the molecule and basis functions---
class contracted_gaussians():
    def __init__(self, alpha, d, coords): #have not yet added angular momenta terms, will have to do for p orbitals etc
        self.alpha = alpha 
        self.d = d #contraction coefficient
        self.coords = coords
        self.Normalisation = (2*alpha/np.pi)**0.75

atomic_coordinates = [np.array([0,0,0])] #Add the coordinates of each atom, using known bond lengths
atomic_masses = [2] #Add the masses of each atom, making sure they're in the same order as above
assert len(atomic_coordinates) == len(atomic_masses)

He_cg1a = contracted_gaussians(0.6362421394E+01, 0.1543289673E+00, atomic_coordinates[0])
He_cg1b = contracted_gaussians(0.1158922999E+01, 0.5353281423E+00, atomic_coordinates[0])
He_cg1c = contracted_gaussians(0.3136497915E+00, 0.4446345422E+00, atomic_coordinates[0])

Hes = [He_cg1a, He_cg1b, He_cg1c]
Slater_bases = [Hes] 
nelectrons = 2 # Number of electrons in the system
nbasis = len(Slater_bases) #Number of basis sets
#------

def E_0(H, P, F):
    E = 0
    for i in range(nbasis):
        for j in range(nbasis):
            E += P[j][i] * (H[i][j] + F[i][j]) / 2
    return E


def SCF_cycle(H, V, S_inverse_sqrt):
    '''
    Calculates a self-consistent set of coefficients
    '''
    P=np.zeros((nbasis,nbasis)) # initial guess for density matrix
    max_cycles = 10 #Maximum cycles to prevent n infinite loop
    C_conv=10**(-6) #convergence criterion
    for i in range(max_cycles):
        Old_P = P 
        G = mat.G_matrix(P, V, nbasis)
        F = H + G #Calculate Fock matrix 
        F_diagonalised = np.dot(S_inverse_sqrt, np.dot(F, S_inverse_sqrt)) #Diagonalise Fock matrix
        C_eigenvalues, C_eigenvectors = np.linalg.eigh(F_diagonalised) #Calculate C' eigeneverything
        C = np.dot(S_inverse_sqrt, C_eigenvectors)
        P = mat.density_matrix(C, nbasis, nelectrons)
        if np.max(np.abs(P - Old_P)) < C_conv:
            print('Convergence criterion satisfied')
            return P, F
        else:
            i +=1
    return P

#---Define the relevant matrices----
H = mat.H_core(Slater_bases, atomic_coordinates, atomic_masses, nbasis) #Core hamiltonian
V = mat.V_ee(Slater_bases, nbasis) #Electron-electron repulsion
S = mat.Overlap_matrix(Slater_bases, nbasis) #Overlap matrix
S_inverse_sqrt = linalg.sqrtm(linalg.inv(S))
#------

#Call the SCF cycle, calculate electronic energy
Density_matrix, Fock_matrix = SCF_cycle(H, V, S_inverse_sqrt)
Electronic_energy = E_0(H, Density_matrix, Fock_matrix)
print(Electronic_energy)