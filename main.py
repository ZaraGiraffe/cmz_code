import matplotlib.pyplot as plt
import numpy as np
import math

eps = 0.0001

def f(x):
    return x**3 - 3 * x**2 - 14 * x - 8

def fi(x):
    return abs(3 * x**2 + 14 * x + 8)**(1.0/3.0) * np.sign(3 * x**2 + 14 * x + 8)

def fis(x):
    return 1.0/3.0*abs(3 * x**2 + 14 * x + 8)**(-2.0/3.0) * np.sign(3 * x**2 + 14 * x + 8) * (6*x+14)

def fi2(x):
    return abs(17*x+7)**(1.0/3.0) * np.sign(17*x+7) + 1

def fi2s(x):
    return 17.0/3.0*abs(17*x+7)**(-2.0/3.0) * np.sign(17*x+7) 

def fi3(x):
    return 3 + 14.0 / x + 8.0 / x**2

def fi3s(x):
    return -14.0 / x**2 -14.0 / x**3

def fi4(x):
    return 1.0/14.0*(x**3 - 3*x**2 - 8)

def fi4s(x):
    return 1.0/14.0*(3*x**2 - 6*x)

def fs(x):
    return 3 * x**2 - 6 * x - 14

def fss(x):
    return 6 * x - 6

def print_func(ax, f, a, b, num=100):
    arr = np.linspace(a, b, num)
    ax.plot(arr, f(arr))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

def dihotomy(f, a, b, maxi=20, e=eps):
    cnt = 0
    while b - a > e and cnt < maxi:
        m = (a + b) / 2.0
        if f(m) * f(a) > 0: a = m
        else: b = m
        cnt += 1
        print(f'iteration {cnt}: x = {m}')
    return (a + b) / 2.0, cnt != maxi

def iteration(f, fi, x0, maxi=50, e=eps):
    cnt = 0
    print(f'iteration {cnt+1}: x = {x0}')
    x1 = fi(x0)
    print(f'iteration {cnt+2}: x = {x1}')
    while f(x1) * f(x1 + e) > 0 and f(x1) * f(x1 - e) > 0 and cnt < maxi:
        x0, x1 = x1, fi(x1)
        cnt += 1
        print(f'iteration {cnt+2}: x = {x1}')

    return x1, cnt != maxi

def newton(f, fs, x, maxi=20, e=eps):
    def func(x):
       return x - f(x) / fs(x)
    cnt = 0
    while f(x) * f(x + e) > 0 and f(x) * f(x - e) > 0 and cnt < maxi:
        print(f'iteration {cnt+1}: x = {x}')
        x = func(x)
        cnt += 1
    print(f'iteration {cnt+1}: x = {x}')
    return x, cnt != maxi
