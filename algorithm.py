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

def delta():
    return

while delta > C_conv:
    '''
    get transformed fock matrix
    solve roothaan
    call density func with new coeffitients
    generate new delta
    '''