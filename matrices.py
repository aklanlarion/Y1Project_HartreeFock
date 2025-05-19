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

nbasis = len(alg.Slater_bases)

def H_core(Slater_bases):
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