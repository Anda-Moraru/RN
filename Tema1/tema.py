import pathlib
import re
import math


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A , B = [], []

    with path.open() as file:
        for line in file:
            left, right = line.split('=')

            number = float(right.strip())
            B.append(number)

            coefficients = [0, 0, 0]

            x_match = re.search(r'([+-]?\s*\d*\.?\d*)\s*x', left)
            y_match = re.search(r'([+-]?\s*\d*\.?\d*)\s*y', left)
            z_match = re.search(r'([+-]?\s*\d*\.?\d*)\s*z', left)

            def coefficient(match):
                if match:
                    c = match.group(1).replace(" ", "")
                    if c == '' or c == '+':
                        return 1.0
                    elif c == '-':
                        return -1.0
                    else:
                        return float(c)
                return 0.0

            coefficients[0] = coefficient(x_match)
            coefficients[1] = coefficient(y_match)
            coefficients[2] = coefficient(z_match)

            A.append(coefficients)

    return A, B


def determinant(a: list[list[float]]) -> float:
    if len(a) == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    elif len(a) == 3:
        return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) -
                a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2]) +
                a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))

def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]

def norm(vector: list[float]) -> float:
    sum = 0
    for b in vector:
        sum += b ** 2
    return math.sqrt(sum)

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = [0 for _ in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i] += matrix[i][j] * vector[j]

    return result

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    detA = determinant(matrix)
    if detA == 0:
        raise ValueError("No solution or infinitely many solutions are given.")

    Ax = [row[:] for row in matrix]
    Ay = [row[:] for row in matrix]
    Az = [row[:] for row in matrix]

    for i in range(len(matrix)):
        Ax[i][0] = vector[i]
        Ay[i][1] = vector[i]
        Az[i][2] = vector[i]

    detX = determinant(Ax)
    detY = determinant(Ay)
    detZ = determinant(Az)

    x = detX / detA
    y = detY / detA
    z = detZ / detA

    return [x, y, z]

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactors = []
    for i in range(len(matrix)):
        cofactors_row = []
        for j in range(len(matrix)):
            det_minor = determinant(minor(matrix, i, j))
            cofactors_row.append((-1) ** (i + j) * det_minor)
        cofactors.append(cofactors_row)
    return cofactors

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    cofactors = cofactor(matrix)
    return transpose(cofactors)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    detA = determinant(matrix)

    a = adjoint(matrix)
    inverse_matrix = [[a[i][j] / detA for j in range(len(matrix))] for i in range(len(matrix))]

    return multiply(inverse_matrix, vector)


A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{solve(A, B)=}")
