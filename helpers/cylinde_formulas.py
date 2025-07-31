"""

Get the formula for a radius of a cylinder in terms of volume and surface area.

"""
import sympy as sp
import numpy as np
from scipy.optimize import minimize

# def residuals(x):
#     V0 = 141000
#     A0 = 25000


#     A = x[0]
#     V = x[1]
#     r = x[2]
#     h = x[3]

#     res = np.zeros(4)

#     # keeping all of these linear makes the radius least
#     # important since the surface area affects heat transfer
#     # and the Volume affects mass-pressure relationships!

#     res[0] = A - A0
#     res[1] = V - V0
#     res[2] = (V - np.pi * r**2 * h) # emphasize consistent volume
#     res[3] = A - 2 * np.pi * r * h - 2 * np.pi * r**2

#     return sum(res)**2


# xsol = minimize(residuals,[25000,141000,10,168])

# breakpoint()








# Define the variables
r = sp.symbols('r')
h = sp.symbols('h')

# Define the equation
#eq1 = 2*sp.pi*r**3 - A*r + 2*V

eq1 = 141000 - sp.pi * r ** 2 * h
eq2 = 25000 - 2 * sp.pi * r * h + 2 * sp.pi * r **2


# Solve the equation
solution = sp.solve((eq1, eq2), (r,h), domain=sp.CC)

for sol in solution:
    for tup in sol:
        print(tup.evalf())


breakpoint()

pass
