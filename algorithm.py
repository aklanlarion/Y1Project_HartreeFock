'''
This file containts the code for running the SCF algorithm. The inputs are given matrices & 
functions performing the operations, and its output is a .cube file.
The self-consistent field cycle, definition of the bases, and the idea for the iteration are derived from 
https://github.com/nickelandcopper/HartreeFockPythonProgram/blob/main/Hartree_Fock_Program.ipynb
'''

import numpy as np
from scipy import linalg
import matrices as mat
import intergrals as integrals
import atoms
import matplotlib.pyplot as plt

import sys


def electron_density(points, density_matrix, Slater_bases):
    '''
    Calculate the charge density at an array of x values given a density matrix
    '''
    original_shape = points.shape[1:]
    npoints = np.prod(original_shape)
    points_flat = points.reshape(3, -1).T  # Shape (npoints, 3)
    
    dens_flat = np.zeros(npoints)
    nbasis = len(Slater_bases)
    
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    g1 = Slater_bases[i][k]
                    g2 = Slater_bases[j][l]
                    l1, m1, n1, A, d1, N1, alpha1 = g1.l, g1.m, g1.n, g1.coords, g1.d, g1.Normalisation, g1.alpha
                    l2, m2, n2, B, d2, N2, alpha2 = g2.l, g2.m, g2.n, g2.coords, g2.d, g2.Normalisation, g2.alpha
                    rA_sep = points_flat - A
                    rB_sep = points_flat - B
                    rA_sep2 = np.sum(rA_sep**2, axis=1)
                    rB_sep2 = np.sum(rB_sep**2, axis=1)
                    xA, yA, zA = rA_sep[:, 0], rA_sep[:, 1], rA_sep[:, 2]
                    xB, yB, zB = rB_sep[:, 0], rB_sep[:, 1], rB_sep[:, 2]

                    phi1 = xA**l1 * yA**m1 * zA**n1 * np.exp(-alpha1*rA_sep2)
                    phi2 = xB**l2 * yB**m2 * zB**n2 * np.exp(-alpha2*rB_sep2)
                    N = N1*N2*d1*d2

                    dens_flat += density_matrix[i][j] * N * phi1 * phi2
    return dens_flat.reshape(original_shape)

def E_tot(H, P, F):
    '''
    Calculate the total energy of the system given the core hamiltonian, density matrix, and fock matrix
    Note that this is total and not electronic energy as H_core includes the nuclear-nuclear repulsion.
    '''
    E = integrals.nuclear_repulsion(atomic_masses, atomic_coordinates) #Add nuclear-nuclear repulsion energy to E
    for i in range(nbasis):
        for j in range(nbasis):
            E += P[j][i] * (H[i][j] + F[i][j]) / 2
    return E 


def SCF_cycle(H, V, S_inverse_sqrt):
    '''
    Calculates a self-consistent set of coefficients
    '''
    P=np.zeros((nbasis,nbasis)) # initial guess for density matrix
    max_cycles = 20 #Maximum cycles to prevent n infinite loop
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
    return P, F

#This is where you change between different atoms.
nelectrons = 2 # net number of electrons of the entire system
atomic_masses = [1, 1] # the atomic mass of each component
atomic_coordinates = [np.array([0,0,0]), np.array([1.4, 0, 0])] # define the separation between the atoms in the compound


xval = np.arange(0.01, 3, 0.01)
#x3dval = np.array([[i, 0.0, 0.0] for i in xval]) #smart way to calculate at different seperations
Energy = np.zeros([len(xval)])

assert len(atomic_coordinates) == len(atomic_masses)
    
Slater_bases = atoms.H(atomic_coordinates[0]) + atoms.H(atomic_coordinates[1])# also edit this to change the type of atoms in the compound
nbasis = len(Slater_bases)

S = mat.Overlap_matrix(Slater_bases, nbasis) #Overlap matrix
print('S', S)
T = mat.Kinetic(Slater_bases, nbasis)
print('T', T)
V_ne = mat.Nuclear_electron(Slater_bases, atomic_coordinates, atomic_masses, nbasis)
print('V_ne', V_ne)
H = mat.H_core(Slater_bases, atomic_coordinates, atomic_masses, nbasis) #Core hamiltonian
print('H', H)
V = mat.V_ee(Slater_bases, nbasis) #Electron-electron repulsion
print('V', V)
S_inverse_sqrt = linalg.sqrtm(linalg.inv(S))
Density_matrix, Fock_matrix = SCF_cycle(H, V, S_inverse_sqrt)
print('Density', Density_matrix)
Energy = E_tot(H, Density_matrix, Fock_matrix)
print('Energy', Energy)

'''
Energy[i] = E_tot(H, Density_matrix, Fock_matrix)

plt.plot(xval, Energy)
plt.xlabel('Distance, Angstrom')
plt.ylabel('Energy, Hartrees')
plt.show()
'''


#---Graph charge density----

fig, ax = plt.subplots()

r = np.arange(-1, 1, 0.1)
X, Y, Z = np.meshgrid(r, r, r)
points = np.array([X, Y, Z])
edensity=electron_density(points, Density_matrix, Slater_bases)

'''
ax.plot(xval, edensity,label='Electron density')

plt.xlabel(r'Distance [${\AA}$]')
plt.ylabel(r'Electron Density [$e/Bohr^3$]')
ax.set_title(r'Electron Density of $Na$')

ax.set_yticks(np.arange(0,max(edensity)*1.01,round(max(edensity)*0.25,1)))
ax.set_yticks(np.arange(0,max(edensity)*1.01,max(edensity)*0.025), minor=True)
ax.set_xticks(np.arange(-3,3,1/10), minor=True)
ax.grid(True,alpha=0.4)
ax.set_ylim([min(edensity)-max(edensity)*0.01,max(edensity)+max(edensity)*0.01])
ax.set_xlim([-3,3])
plt.savefig('Na',dpi=300)
plt.show()
'''