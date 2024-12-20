import numpy as np

from metodenumerice import data
from metodenumerice.jacobi import jacobi
from metodenumerice.jor import jor
from metodenumerice.gs import gs
from metodenumerice.gsb import gsb


if __name__ == '__main__':
    x0 = np.zeros_like(data.b1)

    x_jacobi = jacobi(data.A3, data.b3, x0)
    print("Jacobi solution: ", x_jacobi)
    print("\n")

    x_jor = jor(data.A3, data.b3, x0, omega=0.8)
    print("JOR solution: ", x_jor)
    print("\n")

    x_gs = gs(data.A2, data.b2, x0)
    print("GS solution: ", x_gs)
    print("\n")

    x_gsb = gsb(data.A3, data.b3, x0)
    print("GSb solution: ", x_gsb)
    print("\n")
