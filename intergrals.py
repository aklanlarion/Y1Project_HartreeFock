'''
This file contains the integrals and forms the matrices from the input values
'''

#necessary packages
import sys 
import numpy as np
from scipy import special
from scipy import linalg

#input files - basis, num of electrons, separation, etc

def gauss_product(Gaussian_1, Gaussian_2):
    # Most of the integrals involve taking the product between two Gaussians.
    # Two unnormalised gaussians g(r-R_a) and g(r-R_b) combine to give Kg(r-R_p)
    # Where K, the proportionality constant, is defined as below. See pp410 Modern Quantum Chemistry
    alpha, d_a, R_a, N_a = Gaussian_1.alpha, Gaussian_1.d, Gaussian_1.coords, Gaussian_1.Normalisation
    beta, d_b, R_b, N_b = Gaussian_2.alpha, Gaussian_2.d, Gaussian_2.coords, Gaussian_2.Normalisation
    p = alpha + beta
    R_p = (alpha * R_a + beta * R_b) / p
    R_sep = np.linalg.norm(R_a - R_b)**2
    K = np.exp( -alpha*beta/p * R_sep)
    N = N_a * N_b * d_a * d_b
    return p, R_p, R_sep, K, N

def one_electron_kinetic(Gaussian_1, Gaussian_2):
    '''
    This function calculates the one-electron integral for the kinetic energy of the electron (-1/2*laplace) and returns its value.
    Inputs: r,s - indeces of the basis functions being integrated.
    '''
    alpha, d_a, R_a, N_a = Gaussian_1.alpha, Gaussian_1.d, Gaussian_1.coords, Gaussian_1.Normalisation
    beta, d_b, R_b, N_b = Gaussian_2.alpha, Gaussian_2.d, Gaussian_2.coords, Gaussian_2.Normalisation
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    Term1 = alpha * beta / p
    Term2 = 3 - 2*alpha*beta/p * R_sep
    Term3 = (np.pi/p)**1.5
    return N*Term1*Term2*Term3*K

def boys(t):
    # This variant of the boys function is needed to evaluate nuclear-electron potential
    if t == 0:
        return 1
    else:
        return 0.5 * (np.pi/t)**0.5 * special.erf(t**0.5)

def one_electron_potential(Gaussian_1, Gaussian_2, R_c, Z):
    '''
    This is function calculates the one-electron integral for the potential term (Z/r) and returns its numerical value.
    Inputs are the two contracted gaussians, and the atomic mass and coordinate it is currently being interacted with
    '''
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    Term1 = -2*np.pi/p*Z
    boys_input = p * np.linalg.norm(R_p - R_c)**2
    Term2 = boys(boys_input)
    return N*Term1*K*Term2

def two_electron(Gaussian_1, Gaussian_2, Gaussian_3, Gaussian_4):
    '''
    Calculates the two-electron integral and returns its value (psi psi 1/r psi psi)
    '''
    p, R_p, R_ab, K_ab, N_ab = gauss_product(Gaussian_1, Gaussian_2)
    q, R_q, R_cd, K_cd, N_cd = gauss_product(Gaussian_3, Gaussian_4)
    Term1 = 2*np.pi**2.5 / (p * q * np.sqrt(p+q))
    Term2 = K_ab*K_cd
    boys_input = p*q / (p+q) * np.linalg.norm(R_p-R_q)**2
    Term3 = boys(boys_input)
    N = N_ab*N_cd
    return N * Term1 * Term2 * Term3

def overlap(Gaussian_1,Gaussian_2):
    '''
    Returns the overlap matrix of the basis set
    input: m, n - indeces of functions
    '''
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    return N*(np.pi/p)**1.5*K

def nuclear_repulsion(atomic_masses, atomic_coordinates):
    natoms = len(atomic_masses)
    E_nn = 0 
    if natoms == 1:
        return 0
    else:
        for i in range(natoms):
            for j in range(i+1, natoms):
                Rij = np.linalg.norm(atomic_coordinates[i] - atomic_coordinates[j])
                E_nn += atomic_masses[i]*atomic_masses[j]/Rij
        return E_nn

#wrirte all teh values out for each integral into a file