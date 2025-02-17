import numpy as np

# a)
A1 = np.array([[ 3, 0, 4],
               [ 7, 4, 2],
               [-1, 1, 2]])
b1 = np.array([7, 13, 2])

# b)
A2 = np.array([[ 3, 3, -6],
               [-4, 7, -8],
               [ 5, 7, -9]])
b2 = np.array([0, -5, 3])

# c)
A3 = np.array([[ 4,  1,  1],
               [ 2, -9,  0],
               [ 0, -8, -6]])
b3 = np.array([6, -7, -14])

# d)
A4 = np.array([[ 3, -1,  0],
               [-1,  3, -1],
               [ 0, -1,  3]])
b4 = np.array([2, 1, 2])

# e)
A5 = np.zeros([80, 80])

for i in range(80):
    A5[i, i] = 2*(i+1)
    if i >= 2:
        A5[i, i-2] = i/2
    if i >= 4:
        A5[i, i-4] = i/4

for i in range(78):
    A5[i, i+2] = i/2
    if i <= 75:
        A5[i, i+4] = i/4

b5 = np.full((80,), np.pi)

# SPD Matrix
A6 = np.array([[3, -1, 0],
              [-1, 3, -1],
              [0, -1, 3]], dtype=float)
b6 = np.array([2, 1, 2], dtype=float)
