from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from main import print_func
from main import dihotomy
from main import eps


def newton_interpolation(X, Y):
    n = len(X)
    mas = deepcopy(Y)
    fx = [0.0, Y[0]]
    for i in range(1, n):
        mas2 = []
        for j in range(n-i):
            ny = (mas[j+1] - mas[j]) / (X[j+i] - X[j])
            mas2.append(ny)
        mas = mas2
        fx.append(mas[0])
    res = np.poly1d([0])
    for i in range(n):
        add = np.poly1d(fx[i+1])
        for j in range(i):
            add = add * np.poly1d([1, -X[j]])
        res = res + add
    return res


@np.vectorize
def f(x):
    return x**3 + 4*x - 6 + np.cos(x)


@np.vectorize
def ober(x):
    l, r = -4, 4
    eps = 0.0000000000001
    while r - l > eps:
        m = (l + r) / 2
        if f(m) > x:
            r = m
        else:
            l = m
    return (l + r) / 2


def derivative(g, x, n):
    eps = 0.000001
    mas = []
    for i in range(n+1):
        mas.append(g(x+i*eps))
    while len(mas) > 1:
        mas2 = []
        for i in range(1, len(mas)):
            mas2.append((mas[i] - mas[i-1]) / eps)
        mas = mas2
    return mas2[0]

def implicit3_derivative(x):
    return ((-np.sin(x) - 6) + 3 * (6 * x - np.cos(x))) / (3 * x**2 + 4 - np.sin(x))**5

if __name__ == "__main__":
    X = np.linspace(0, 2, 10)
    Y = f(X)
    
    def newton():
        pol = newton_interpolation(X, Y)
        print(pol)
        res1, _ = dihotomy(pol, 0, 2)
        M = 1
        m1 = 3
        w = np.poly1d(1)
        for i in X:
            w = w * np.poly1d([1, -i])
        poh = eps + M * abs(w(res1)) / (m1 * np.math.factorial(11)) 
        print(res1, poh)


        ober_pol = newton_interpolation(Y, X)
        print(ober_pol)
        res2 = ober_pol(0)
        print(res2)

        ax = plt.subplot()
        plt.scatter(ober_pol(np.linspace(f(0), f(2), 100)), np.linspace(f(0), f(2), 100), c='r', s=10)
        print_func(ax, f, 0, 2)
        plt.show()

    @np.vectorize
    def spline(x):
        n = X.shape[0]
        fmas = Y.reshape(n, 1)
        A = np.zeros((n-2, n-2))
        for i in range(n-2):
            if i > 0:
                A[i, i-1] = (X[i+1] - X[i]) / 6
            if i < n-3:
                A[i, i+1] = (X[i+2] - X[i+1]) / 6
            A[i, i] = (X[i+2] - X[i]) / 3
        H = np.zeros((n-2, n))
        for i in range(n-2):
            H[i, i] = 1 / (X[i+1] - X[i])
            H[i, i+1] = -1 / (X[i+1] - X[i]) - 1 / (X[i+2] - X[i+1])
            H[i, i+2] = 1 / (X[i+2] - X[i+1])
        A1 = np.linalg.inv(A)
        m = np.matmul(A1, np.matmul(H, fmas)).flatten()
        if x < X[0] or x > X[-1]:
            return 0
        for i in range(1, n):
            if X[i] > x:
                hi = X[i] - X[i-1]
                mi = 0 if i == n-1 else m[i-1]
                mi1 = 0 if i == 1 else m[i-2]
                fi = Y[i]
                fi1 = Y[i-1]
                Bi = fi - mi * hi**2 / 6
                Ai = fi1 - mi1 * hi**2 / 6
                s1 = np.poly1d([1, -X[i]])**3 * (mi1 / (6 * hi))
                s2 = np.poly1d([1, -X[i-1]])**3 * (mi / (6 * hi))
                s3 = np.poly1d([-1, X[i]]) * (Ai / hi)
                s4 = np.poly1d([1, -X[i-1]]) * (Bi / hi)
                s = s1 + s2 + s3 + s4
                return s(x)
    
    #newton()
    
    ax = plt.subplot()
    print_func(ax, f, 0, 2)
    ax.scatter(np.linspace(0, 2, 100), spline(np.linspace(0, 2, 100)), c='r', s=10)
    plt.show()

