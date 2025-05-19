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
    alpha, d_a, R_a, N_a = Gaussian_1
    beta, d_b, R_b, N_b = Gaussian_2
    p = alpha + beta
    R_p = (alpha * R_a + beta * R_b) / p
    R_sep = np.linalg.norm(R_a - R_b)**2
    K = np.exp( -alpha*beta/p * R_sep)
    return p, R_p, R_sep, K

def one_electron_kinetic(Gaussian_1, Gaussian_2):
    '''
    This function calculates the one-electron integral for the kinetic energy of the electron (-1/2*laplace) and returns its value.
    Inputs: r,s - indeces of the basis functions being integrated.
    '''
    alpha, d_a, R_a, N_a = Gaussian_1
    beta, d_b, R_b, N_b = Gaussian_2
    p, R_p, R_sep, K = gauss_product(Gaussian_1, Gaussian_2)
    N = N_a * N_b * d_a * d_b
    coeff1 = alpha * beta / p
    coeff2 = 3 - 2*alpha*beta/p * R_sep
    coeff3 = (np.pi/alpha+beta)**3/2
    return N*coeff1*coeff2*coeff3*K

def one_electron_potential(Gaussian_1, Gaussian_2):
    '''
    This is function calculates the one-electron integral for the potential term (Z/r) and returns its numerical value.
    Inputs: r,s - indeces of the basis functions
    '''
    return

def two_electron(r,s,t,u):
    '''
    Calculates the two-electron integral and returns its value (psi psi 1/r psi psi)
    input: r,s,t,u - indeces of the four basis funcs your integrating over
    '''
    return

def overlap(m,n):
    '''
    Returns the overlap matrix of the basis set
    input: m, n - indeces of functions
    '''
    return

#wrirte all teh values out for each integral into a file