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
    alpha, d, R_a, Normalisation = Gaussian_1
    beta, d, R_b, Normalisation = Gaussian_2
    p = alpha + beta
    R_p = (alpha * R_a + beta * R_b) / p
    R_sep = np.linalg.norm(R_a - R_b)**2
    K = np.exp( -alpha*beta/p * R_sep)
    return p, R_p, R_sep, K

def one_electron_kinetic(r, s):
    '''
    This function calculates the one-electron integral for the kinetic energy of the electron (-1/2*laplace) and returns its value.
    Inputs: r,s - indeces of the basis functions being integrated.
    '''
    return

def one_electron_potential(r,s):
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