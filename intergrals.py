'''
This file contains the integrals and forms the matrices from the input values
'''

#necessary packages
import sys 
import numpy as np
from scipy import special
from scipy import linalg
from functools import lru_cache
import math
import atoms


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
        print('NEGATIVE CHECK KINETIC')
        sys.exit
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

# Memoized Boys function (or replace with a fast approximation)
@lru_cache(maxsize=None)

def hermite_integral(p, X_pc, Y_pc, Z_pc, R_cp, t_max, u_max, v_max, n_max):
    # Initialize DP table: dp[t][u][v][n]
    #N.b. X, Y, Z, R are cartesian DISTANCES; p = alpha+beta
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

def differential_2(pos_j, g1, g2):
    #Calculate the second differential of two gaussians. Used for kinetic energy calculation.
    beta, L = g2.alpha, g2.L
    j = L[pos_j]
    L1, L3 = L.copy(), L.copy()
    L1[pos_j] += 2
    L3[pos_j] = max(L3[pos_j] - 2, 0)
    print(L1, L3)
    Gaussian_T1 = atoms.contracted_gaussians(g2.alpha, g2.d, g2.coords, L1)
    Gaussian_T3 = atoms.contracted_gaussians(g2.alpha, g2.d, g2.coords, L3)
    Term1 = j*(j-1)*overlap(g1, Gaussian_T1)[pos_j + 1]
    Term2 = -2*beta*(2*j+1)*overlap(g1, g2)[pos_j + 1]
    Term3 = 4*beta**2*overlap(g1, Gaussian_T3)[pos_j + 1]
    return Term1 + Term2 + Term3

def temp_gaussian(pos_j, g):
    L = g.L
    Lplus2, Lminus2 = L.copy(), L.copy()
    Lplus2[pos_j] +=2
    Lminus2[pos_j] = max(Lminus2[pos_j] - 2, 0) 
    return atoms.contracted_gaussians(g.alpha, g.d, g.coords, Lplus2), atoms.contracted_gaussians(g.alpha, g.d, g.coords, Lminus2)

def T(g1, g2):
    l1, m1, n1, N1, d1 = g1.l, g1.m, g1.n, g1.Normalisation, g1.d
    l2, m2, n2, N2, d2, beta = g2.l, g2.m, g2.n, g2.Normalisation, g2.d, g2.alpha

    g2_plus2_0, g2_minus2_0 = temp_gaussian(0, g2)
    g2_plus2_1, g2_minus2_1 = temp_gaussian(1, g2)
    g2_plus2_2, g2_minus2_2 = temp_gaussian(2, g2)

    N = N1 * d1 * N2* d2    
    Term1 = beta * (2*(l2+m2+n2)+3) * overlap(g1, g2)[1]
    Term2 = -2*beta**2 * (overlap(g1, g2_plus2_0)[1] + \
                                       overlap(g1, g2_plus2_1)[1] + \
                                        overlap(g1, g2_plus2_2)[1])
    Term3 = -0.5*((l2*(l2-1)) * overlap(g1, g2_minus2_0)[1] + \
    m2*(m2-1) * overlap(g1, g2_minus2_1)[1] + \
        n2*(n2-1)*overlap(g1, g2_minus2_2)[1])
    return N*(Term1 + Term2 + Term3)




def V(g1, g2, r_n, Z):
    p, R_p, R_sep, K, N = gauss_product(g1, g2)
    R = R_p - r_n
    R_cp = np.linalg.norm(R)
    l1, m1, n1, alpha, A = g1.l, g1.m, g1.n, g1.alpha, g1.coords
    l2, m2, n2, beta, B = g2.l, g2.m, g2.n, g2.alpha, g2.coords
    AB = A-B

    V = 0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                V += hermite_polynomial(alpha, beta, AB[0], l1, l2, t) * \
                hermite_polynomial(alpha, beta, AB[1], m1, m2, u) * \
                hermite_polynomial(alpha, beta, AB[2], n1, n2, v) * \
                hermite_integral(p, R[0], R[1], R[2], R_cp, t, u, v, 0)
    return -Z * V * 2*np.pi/p * N * K

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
    l1, m1, n1, A, alpha, N1, d1 = Gaussian_1.l, Gaussian_1.m, Gaussian_1.n, Gaussian_1.coords, Gaussian_1.alpha, Gaussian_1.Normalisation, Gaussian_1.d
    l2, m2, n2, B, beta, N2, d2 = Gaussian_2.l, Gaussian_2.m, Gaussian_2.n, Gaussian_2.coords, Gaussian_2.alpha, Gaussian_2.Normalisation, Gaussian_2.d
    p = alpha + beta
    S1 = hermite_polynomial(alpha, beta, A[0] - B[0], l1, l2, 0) * np.sqrt(np.pi/p)
    S2 = hermite_polynomial(alpha, beta, A[1] - B[1], m1, m2, 0) * np.sqrt(np.pi/p)
    S3 = hermite_polynomial(alpha, beta, A[2]-B[2], n1, n2, 0) * np.sqrt(np.pi/p)
    E = S1*S2*S3
    N = N1*N2*d1*d2
    return [N*E, E]

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