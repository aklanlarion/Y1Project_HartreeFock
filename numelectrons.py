import algorithm as alg
import numpy as np
from scipy import integrate as integ
from scipy import interpolate as inter


def trapezium_integrate(xarray, yarray):
    return 0.5*np.sum((xarray[1:]-xarray[:-1])*(yarray[1:]+yarray[:-1]))

def edensityy(x,y,z):
    return inter.interpn(alg.points2,alg.edensity,(x,y,z))

print(edensityy(0,0,0))
Enum = integ.tplquad(edensityy,-0.3,0.3,-0.3,0.3,-0.3,0.3,epsabs=1)


#print('Number of electrons in the vicinity of He+: ', He_enum)
#print('Number of electrons in the vicinity of H: ', H_enum)
print('Number of electrons in the system: ', Enum)

