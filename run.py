import numpy as np

from metodenumerice import data
from metodenumerice.jacobi import jacobi
from metodenumerice.jor import jor
from metodenumerice.gs import gs
from metodenumerice.gsb import gsb
from metodenumerice.sor import sor


if __name__ == '__main__':
    x0 = np.zeros_like(data.b1)

    print("==Jacobi==\n")
    c_jacobi = jacobi(data.A3, data.b3, x0)
    print("Solution for c): ", c_jacobi)
    d_jacobi = jacobi(data.A4, data.b4, x0)
    print("Solution for d): ", d_jacobi)
    e_jacobi = jacobi(data.A5, data.b5, np.zeros_like(data.b5))
    print("Solution for e): ", e_jacobi)
    print("\n")

    print("==JOR==\n")
    c_jor = jor(data.A3, data.b3, x0, omega=0.8)
    print("Solution for c): ", c_jor)
    d_jor = jor(data.A4, data.b4, x0, omega=0.8)
    print("Solution for d): ", d_jor)
    print("\n")

    print("==GS==\n")
    x_gs = gs(data.A2, data.b2, x0, opt=1)
    print("Solution for c): ", x_gs)
    print("\n")

    print("==GSb==\n")
    c_gsb = gsb(data.A3, data.b3, x0, opt=1)
    print("Solution for c): ", c_gsb)
    d_gsb = gsb(data.A4, data.b4, x0, opt=1)
    print("Solution for d): ", d_gsb)
    print("\n")

    print("==SOR==\n")
    c_sor = sor(data.A3, data.b3, x0, omega=1.5)
    print("Solution for c): ", c_sor)
    d_sor = sor(data.A4, data.b4, x0, omega=1.5)
    print("Solution for d): ", d_sor)
    print("\n")
