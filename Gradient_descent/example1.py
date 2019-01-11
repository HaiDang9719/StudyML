# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt
def grad(x):
    return -2*np.sin(x)*np.sin(2*x)+ np.cos(x)*np.cos(2*x)+2*x

def cost(x):
    return x**2 + np.cos(2*x)*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-4:
            break
        x.append(x_new)
    return (x, it)
(x1, it1) = myGD1(.01, -5)
(x2, it2) = myGD1(.01, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))