'''
This file containts the matrix operations used in during the Hartree-Fock method. Its inputs are the two- and one-electron integrals, 
and outputs functions containing the matrix operations.
'''

#necessary libraries
import sys
import numpy as np
from scipy import linalg

#import the integrals
import intergrals as int
import algorithm as alg

class contracted_gaussians():
    def __init__(self, alpha, d, coords): #have not yet added angular momenta terms, will have to do for p orbitals etc
        self.alpha = alpha 
        self.d = d #contraction coefficient
        self.coords = coords
        self.Normalisation = (2*alpha/np.pi)**0.75

atomic_coordinates = [np.array([0,0,0])]
atomic_masses = [1]
assert len(atomic_coordinates) == len(atomic_masses)

He_cg1a = contracted_gaussians(0.6362421394E+01, 0.1543289673E+00, atomic_coordinates[0])
He_cg1b = contracted_gaussians(0.1158922999E+01, 0.5353281423E+00, atomic_coordinates[0])
He_cg1c = contracted_gaussians(0.3136497915E+00, 0.4446345422E+00, atomic_coordinates[0])

Hes = [He_cg1a, He_cg1b, He_cg1c]
Slater_bases = [Hes]
nbasis = len(Slater_bases)

def H_core(Slater_bases): #Kinetic and Nuclear-Electron Potential
    H_core = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    H_core[i,j] += int.one_electron_kinetic(Slater_bases[i][k], Slater_bases[j][l]) + int.one_electron_potential(Slater_bases[i][k], Slater_bases[j][l])

    return H_core

def Overlap_matrix():
    return

def Transformation_matrix():
    return

def G_matrix():
    return

def Fock_matrx(H_core, G_matrix):
    return H_core + G_matrix

def Roothaan():
    return

def coeff_matrix():
    return

def Density_matrix():
    return