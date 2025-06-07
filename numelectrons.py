import algorithm as alg
import numpy as np
from scipy import integrate as integ


def trapezium_integrate(xarray, yarray):
    return 0.5*np.sum((xarray[1:]-xarray[:-1])*(yarray[1:]+yarray[:-1]))

xval = np.arange(-3, 3, 0.01)
points = np.array([[i, 0.0, 0.0] for i in xval])
edensityr=alg.electron_density(points, alg.Density_matrix, alg.Slater_bases)*xval**2

middle=int(np.ceil(len(edensityr)/2))
H_enum = trapezium_integrate(alg.xval[:middle],edensityr[:middle])
He_enum = trapezium_integrate(alg.xval[middle:],edensityr[middle:])



Enum = trapezium_integrate(alg.xval,edensityr)


print('Number of electrons in the vicinity of He+: ', He_enum)
print('Number of electrons in the vicinity of H: ', H_enum)
print('Number of electrons in the system: ', Enum)

