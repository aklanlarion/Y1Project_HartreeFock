'''
This file contains the integrals and forms the matrices from the input values
'''

#necessary packages
import sys 
import numpy as np
from scipy import special

#input files - basis, num of electrons, separation, etc

def gauss_product(A, B):
    # Most of the integrals involve taking the product between two Gaussians.

    return

def one_electron_kinetic(r,s):
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