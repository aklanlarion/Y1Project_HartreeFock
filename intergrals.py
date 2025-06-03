'''
This file contains the integrals and forms the matrices from the input values
'''

#necessary packages
import sys 
import numpy as np
from scipy import special
from scipy import linalg
from functools import lru_cache

import sys
sys.setrecursionlimit(1000000)  # default is around 1000


#input files - basis, num of electrons, separation, etc

def gauss_product(Gaussian_1, Gaussian_2):
    # Most of the integrals involve taking the product between two Gaussians.
    # Two unnormalised gaussians g(r-R_a) and g(r-R_b) combine to give Kg(r-R_p)
    # Where K, the proportionality constant, is defined as below. See pp410 Modern Quantum Chemistry
    alpha, d_a, R_a, N_a = Gaussian_1.alpha, Gaussian_1.d, Gaussian_1.coords, Gaussian_1.Normalisation
    beta, d_b, R_b, N_b = Gaussian_2.alpha, Gaussian_2.d, Gaussian_2.coords, Gaussian_2.Normalisation
    p = alpha + beta
    R_p = (alpha * R_a + beta * R_b) / p
    R_sep = np.dot(R_a-R_b, R_a-R_b)
    K = np.exp( -alpha*beta/p * R_sep)
    N = N_a * N_b * d_a * d_b
    return p, R_p, R_sep, K, N

def hermite_polynomial(alpha, beta, dist, i, j, t):
    if i<0 or j<0: 
        print('Negative')
        return 0
    p = alpha + beta
    q = alpha*beta / p
    R_sep = dist**2
    if (t<0 or t>(i+j)):
        return 0
    elif i == j == t == 0:
        return np.exp(-q*R_sep)
    elif j == 0:
        return (1/(2*p))*hermite_polynomial(alpha, beta, dist, i-1, j, t-1) - \
            (q*R_sep/alpha)*hermite_polynomial(alpha, beta, dist, i-1, j, t) + \
            (t+1) * hermite_polynomial(alpha, beta, dist, i-1, j, t+1)
    else:
        return (1/(2*p))*hermite_polynomial(alpha, beta, dist, i, j-1, t-1) - \
            (q*R_sep/beta)*hermite_polynomial(alpha, beta, dist, i, j-1, t) + \
            (t+1) * hermite_polynomial(alpha, beta, dist, i, j-1, t+1)

def boys(n, T):
    return special.hyp1f1(n+0.5, n+1.5, -T)/(2*n+1)

import math
from functools import lru_cache

# Memoized Boys function (or replace with a fast approximation)
@lru_cache(maxsize=None)

def hermite_integral(p, X_pc, Y_pc, Z_pc, R_cp, t_max, u_max, v_max, n_max):
    # Initialize DP table: dp[t][u][v][n]
    dp = {}

    # Precompute Boys function values for all needed n
    boys_cache = {}
    pR2 = p * R_cp**2
    for n in range(n_max + t_max + u_max + v_max + 1):
        boys_cache[n] = boys(n, pR2)

    # Initialize base case (t=0, u=0, v=0) for all possible n
    for n in range(n_max + t_max + u_max + v_max + 1):
        dp[(0, 0, 0, n)] = (-2 * p)**n * boys_cache[n]

    # Iterate over all possible t, u, v in increasing order
    for t in range(t_max + 1):
        for u in range(u_max + 1):
            for v in range(v_max + 1):
                for n in range(n_max + 1):
                    if (t, u, v, n) in dp:
                        continue  # Already computed

                    # Apply recurrence relations
                    res = 0.0
                    if t == 0 and u == 0:
                        if v > 0:
                            if v >= 2:
                                res += (v - 1) * dp.get((t, u, v - 2, n + 1), 0)
                            res += Z_pc * dp.get((t, u, v - 1, n + 1), 0)
                    elif t == 0:
                        if u > 0:
                            if u >= 2:
                                res += (u - 1) * dp.get((t, u - 2, v, n + 1), 0)
                            res += Y_pc * dp.get((t, u - 1, v, n + 1), 0)
                    else:
                        if t > 0:
                            if t >= 2:
                                res += (t - 1) * dp.get((t - 2, u, v, n + 1), 0)
                            res += X_pc * dp.get((t - 1, u, v, n + 1), 0)

                    dp[(t, u, v, n)] = res

    return dp[(t_max, u_max, v_max, n_max)]


def T_ss(Gaussian_1, Gaussian_2):
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

def T_sp(g1, g2):
    L1, L2 = g1.L, g2.L
    if L1 == [0, 0, 0]:
        gs, gp = g1, g2
    else:
        gs, gp = g2, g1
        
    
    alpha1, alpha2 = gs.alpha, gp.alpha
    r1, r2 = np.array(gs.coords), np.array(gp.coords)   
    N1, N2  = gs.d, gp.d  #contraction coeff
    
#define parameters
    gamma = alpha1 + alpha2 #sum of exponents, same as p in other code
    mu = alpha1*alpha2 / gamma   # mu factor
    P = (alpha1*r1 + alpha2*r2) / gamma  #weighted coordinate (analogues to center of mass)
    c1, c2 = ((2*alpha1)/np.pi) ** 0.75, 2**1.75 * alpha2**1.25 * np.pi**(-0.75)   #normalization factors
    dstc_sqr = np.dot( r1-r2, r1-r2 ) # distance squared
    S = (np.pi/gamma)**1.5 * np.exp(-mu * dstc_sqr) # s-s overlap, it's not two s functions but this term simplifies the expression
    r1_d, r2_d = P - r1, P - r2

    ort = int(np.where(np.array(gp.L) == 1)[0][0]) # orientation of p-orbital

    return (N1*N2*c1*c2) * (2*alpha2/gamma) * r2_d[ort] * S

def T_pp(g1, g2):
    L1, L2 = np.array(g1.L), np.array(g2.L)
    ort1, ort2 = int(np.where(L1 == 1)[0][0]) , int(np.where(L2 == 1)[0][0])

    alpha1, alpha2 = g1.alpha, g2.alpha
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff

    gamma = alpha1 + alpha2 #sum of exponents, same as p in other code
    mu = alpha1*alpha2 / gamma   # mu factor
    P = (alpha1*r1 + alpha2*r2) / gamma  # center of product gaussian
    c1, c2 = 2**1.75 * alpha1**1.25 * np.pi**(-0.75), 2**1.75 * alpha2**1.25 * np.pi**(-0.75)   #normalization factors
    dstc_sqr = np.dot( r1-r2, r1-r2 ) # distance squared
    S = (np.pi/gamma)**1.5 * np.exp(-mu * dstc_sqr) # s-s overlap, it's not two s functions but this term simplifies the expression
    r1_d, r2_d = P - r1, P - r2  # distance vector to P

    u = 2*alpha2/gamma
    
    if ort1 == ort2:
        T_12 = (((u*r2_d[ort1])**2 - u) * r1_d[ort1]**2 + u) * S
    else:
        T_12 = u**2 * r1_d[ort1] * r1_d[ort2] * r2_d[ort1] * r2_d[ort2] * S

    return T_12 * (N1*N2*c1*c2)


def F0(t): #Boys function of order 0
    # This variant of the boys function is needed to evaluate nuclear-electron potential
    if t == 0:
        return 1
    else:
        return 0.5 * (np.pi/t)**0.5 * special.erf(t**0.5)

def F1(t): #Boys function of order 1
    if t == 0:
        return 1/3

    else:
        return (F0(t) - np.exp(-t)) / (2*t)

def F2(t):
    if t == 0:
        return 0.2
    else:
        return (F1(t) - np.exp(-t)/3) / (2*t)

def V_ss(Gaussian_1, Gaussian_2, R_c, Z):
    '''
    This is function calculates the one-electron integral for the potential term (Z/r) and returns its numerical value.
    Inputs are the two contracted gaussians, and the atomic mass and coordinate it is currently being interacted with
    '''
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    Term1 = -2*np.pi*Z/p
    boys_input = p * np.dot(R_p - R_c, R_p-R_c)
    Term2 = F0(boys_input)
    return N*Term1*K*Term2

def V_sp(g1, g2, r_n, Z):
    L1, L2 = g1.L, g2.L
    if L1 == [0, 0, 0]:
        gs, gp = g1, g2
    else:
        gs, gp = g2, g1

    alpha1, alpha2 = gs.alpha, gp.alpha
    r1, r2 = np.array(gs.coords), np.array(gp.coords)   
    N1, N2  = gs.d, gp.d  #contraction coeff
    
#define parameters
    gamma = alpha1 + alpha2 #sum of exponents, same as p in other code
    mu = alpha1*alpha2 / gamma   # mu factor
    P = (alpha1*r1 + alpha2*r2) / gamma  # center of product gaussian
    c1, c2 = ((2*alpha1)/np.pi) ** 0.75, 2**1.75 * alpha2**1.25 * np.pi**(-0.75)   #normalization factors
    dstc_sqr = np.dot( r1-r2, r1-r2 ) # distance squared
    K = np.exp(-mu * dstc_sqr) # s-s overlap, it's not two s functions but this term simplifies the expression
    r_n = np.array(r_n)
    r1_d, r2_d, rn_d = P - r1, P - r2, P - r_n   # distace vectors from the center of product gaussian


    T = gamma * np.dot(rn_d, rn_d) #boys input
    ort = int(np.where(np.array(gp.L) == 1)[0][0]) # orientation of p-orbital
    C = -Z*2*np.pi*K/gamma  #prefactor
    V_12 = C * (r2_d[ort]*F0(T) + rn_d[ort]*F1(T)/gamma)

    return (N1*N2*c1*c2) * V_12

def V_pp(g1, g2, r_n, Z):
    L1, L2 = np.array(g1.L), np.array(g2.L)
    ort1, ort2 = int(np.where(L1 == 1)[0][0]) , int(np.where(L2 == 1)[0][0])

    alpha1, alpha2 = g1.alpha, g2.alpha
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff

    gamma = alpha1 + alpha2 #sum of exponents, same as p in other code
    mu = alpha1*alpha2 / gamma   # mu factor
    P = (alpha1*r1 + alpha2*r2) / gamma  # center of product gaussian
    c1, c2 = 2**1.75 * alpha1**1.25 * np.pi**(-0.75), 2**1.75 * alpha2**1.25 * np.pi**(-0.75)   #normalization factors
    dstc_sqr = np.dot( r1-r2, r1-r2 ) # distance squared
    K = np.exp(-mu * dstc_sqr) 
    r1_d, r2_d, rn_d = P - r1, P - r2, P - r_n   # distace vectors from the center of product gaussian

    T = gamma * np.dot(rn_d, rn_d) #boys input
    C = -Z*2*np.pi*K/gamma  #prefactor
    
    if ort1 == ort2:
        V_12 = (r1_d[ort1]*r2_d[ort1] + 1/(2*gamma))*F0(T) + (r1_d[ort1] + r2_d[ort1])*(F1(T)*rn_d[ort1]/gamma) + 0.5*((rn_d[ort1]/gamma)**2)*F2(T)
    
    else:
        V_12 = r1_d[ort1]*r2_d[ort2]*F0(T) + r1_d[ort1]*rn_d[ort2]*F1(T)/gamma + r2_d[ort2]*rn_d[ort1]*F1(T)/gamma + rn_d[ort1]*rn_d[ort2]*F2(T)*(gamma**(-2))

    return (C*N1*N2*c1*c2)*V_12

def two_electron(Gaussian_1, Gaussian_2, Gaussian_3, Gaussian_4):
    '''
    Calculates the two-electron integral and returns its value (psi psi 1/r psi psi)
    '''
    #Below, extract p, q, angular coefficients, and coordinates as needed for the calculation
    p, R_p, R_ab, K_ab, N_ab = gauss_product(Gaussian_1, Gaussian_2)
    q, R_q, R_cd, K_cd, N_cd = gauss_product(Gaussian_3, Gaussian_4)
    R_cp = np.linalg.norm(R_p - R_q)
    l1, m1, n1, A, alpha = Gaussian_1.l, Gaussian_1.m, Gaussian_1.n, Gaussian_1.coords, Gaussian_1.alpha
    l2, m2, n2, B, beta = Gaussian_2.l, Gaussian_2.m, Gaussian_2.n, Gaussian_2.coords, Gaussian_2.alpha
    l3, m3, n3, C, gamma = Gaussian_3.l, Gaussian_3.m, Gaussian_3.n, Gaussian_3.coords, Gaussian_3.alpha
    l4, m4, n4, D, delta = Gaussian_4.l, Gaussian_4.m, Gaussian_4.n, Gaussian_4.coords, Gaussian_4.alpha
    
    g = 0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            g += hermite_polynomial(alpha, beta, A[0]-B[0], l1, l2, t) * \
                            hermite_polynomial(alpha, beta, A[1]-B[1], m1, m2, u) * \
                            hermite_polynomial(alpha, beta, A[2] - B[2], n1, n2, v) * \
                            (-1)**(tau+nu+phi) * \
                            hermite_polynomial(gamma, delta, C[0]-D[0], l3, l4, tau) * \
                            hermite_polynomial(gamma, delta, C[1]-D[1], m3, m4, nu) * \
                            hermite_polynomial(gamma, delta, C[2] - D[2], n3, n4, phi) * \
                            hermite_integral((p*q/(p+q)), R_p[0] - R_q[0], R_p[1] - R_q[1], \
                                             R_p[2] - R_q[2], R_cp, t+tau, u+nu, v+phi, 0)

    g *= 2*np.pi**2.5 / (p * q * np.sqrt(p+q))
    g *= N_ab*N_cd
    return g

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


def ss_overlap(g1, g2):
    alpha1, alpha2 = g1.alpha, g2.alpha
    p, q = alpha1 + alpha2, alpha1 * alpha2           #define parameters
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff
    c1, c2 = ((2*alpha1)/np.pi) ** 0.75, ((2*alpha2)/np.pi) ** 0.75   #normalization factors
    
    dstc_sqr = np.dot( r1-r2, r1-r2 )                  #distence squared
    K_12 = np.exp(-(q/p) * dstc_sqr ) #exponential prefactor

    inner_prod = K_12 * (N1*N2*c1*c2) * (np.pi/p)**1.5

    return inner_prod

def sp_overlap(g1, g2):
    alpha1, alpha2 = g1.alpha, g2.alpha
    p, q = alpha1 + alpha2, alpha1 * alpha2           #define parameters
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff
    dstc_sqr = np.dot( r1-r2, r1-r2 )                  #distence squared
    K_12 = np.exp(-(q/p) * dstc_sqr ) #exponential prefactor

    if g1.L == [0, 0, 0] :# if g2 is the p-type gaussian
        c1, c2 = ((2*alpha1)/np.pi) ** 0.75, 2**1.75 * alpha2**1.25 * np.pi**(-0.75) #normalization factors
        if g2.L[0] == 1:#if g2 is px
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha1 * p**(-2.5) * (r1[0] - r2[0]) * np.pi**1.5
        elif g2.L[1] == 1: # if g2 is py
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha1 * p**(-2.5) * (r1[1] - r2[1]) * np.pi**1.5
        else: # if g2 is pz
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha1 * p**(-2.5) * (r1[2] - r2[2]) * np.pi**1.5

    else: #g1 is p-type
        c1, c2 = 2**1.75 * alpha1**1.25 * np.pi**(-0.75), ((2*alpha2)/np.pi) ** 0.75
        if g1.L[0] == 1:#if g1 is px
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha2 * p**(-2.5) * (r2[0] - r1[0]) * np.pi**1.5
        elif g1.L[1] == 1: # if g1 is py
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha2 * p**(-2.5) * (r2[1] - r1[1]) * np.pi**1.5
        else: #if G1 is pz
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha2 * p**(-2.5) * (r2[2] - r1[2]) * np.pi**1.5

    return inner_prod

def pp_overlap(g1, g2):
    alpha1, alpha2 = g1.alpha, g2.alpha
    p, q = alpha1 + alpha2, alpha1 * alpha2           #define parameters
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff
    dstc_sqr = np.dot( r1-r2, r1-r2 )                  #distence squared
    K_12 = np.exp(-(q/p) * dstc_sqr ) #exponential prefactor
    c1, c2 = 2**1.75 * alpha1**1.25 * np.pi**(-0.75), 2**1.75 * alpha2**1.25 * np.pi**(-0.75) #normalization factors
    L1, L2 = g1.L, g2.L

    if L1 == [1, 0, 0] and L2 == [1, 0, 0]:#both px
        inner_prod = K_12 * (N1*N2*c1*c2) * ( 1/(2*p) - q * ((r1[0]-r2[0])/p)**2 ) * (np.pi/p)**1.5
    elif L1 == [0, 1, 0] and L2 == [0, 1, 0]:#both py
        inner_prod = K_12 * (N1*N2*c1*c2) * ( 1/(2*p) - q * ((r1[1]-r2[1])/p)**2 ) * (np.pi/p)**1.5
    elif L1 == [0, 0, 1] and L2 == [0, 0, 1]:#both pz
        inner_prod = K_12 * (N1*N2*c1*c2) * ( 1/(2*p) - q * ((r1[2]-r2[2])/p)**2 ) * (np.pi/p)**1.5
    elif L1 == [1, 0, 0] and L2 == [0, 1, 0]:#g1 px, g2 py
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r2[0] - r1[0]) * (r1[1] - r2[1])
    elif L1 == [1, 0, 0] and L2 == [0, 0, 1]:#g1 px, g2 pz
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r2[0] - r1[0]) * (r1[2] - r2[2])
    elif L1 == [0, 1, 0] and L2 == [1, 0, 0]: #g1 py, g2 px
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[0] - r2[0]) * (r2[1] - r1[1])
    elif L1 == [0, 1, 0] and L2 == [0, 0, 1]: #g1 py, g2 pz
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[2] - r2[2]) * (r2[1] - r1[1])
    elif L1 == [0, 0, 1] and L2 == [1, 0, 0]: #g1 pz, g2 px
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[0] - r2[0]) * (r2[2] - r1[2])
    else: #g1 pz, g2 py
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[1] - r2[1]) * (r2[2] - r1[2])
        
    return inner_prod

#wrirte all teh values out for each integral into a file
