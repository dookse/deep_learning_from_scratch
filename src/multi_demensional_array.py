import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A, B))

A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)

B = np.array([7, 8])
print(B.shape)

C = np.dot(A, B)
print(C)
print(C.shape)
