'''
This file containts the code for running the SCF algorithm. The inputs are given matrices & 
functions performing the operations, and its output is a .cube file. 
'''

#importing necessary packages
import numpy as np

#import the matrices package
import matrices

C_conv=0.1 #convergence criterion
delta = 1 #initial delta
dim=2 # number of basis set
P=np.zeros((dim,dim)) # initial guess for density matrix

def delta():
    return

while delta > C_conv:
    '''
    get transformed fock matrix
    solve roothaan
    call density func with new coeffitients
    generate new delta
    '''