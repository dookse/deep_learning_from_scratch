import numpy as np


def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    return int(tmp > 0)


def nand_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    return int(tmp > 0)


def or_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(x * w) + b
    return int(tmp > 0)


print(and_gate(0, 0))
print(and_gate(1, 0))
print(and_gate(0, 1))
print(and_gate(1, 1), end='\n\n')

print(nand_gate(0, 0))
print(nand_gate(1, 0))
print(nand_gate(0, 1))
print(nand_gate(1, 1), end='\n\n')

print(or_gate(0, 0))
print(or_gate(1, 0))
print(or_gate(0, 1))
print(or_gate(1, 1))
