import numpy as np

def toeq(z, num):
    if len(z) < num:
        return ' ' * (num - len(z)) + z
    else:
        return z[:num]


def print_mat(A, exp=True):
    num = 0
    if exp:
        for i in A.flatten():
            num = max(num, len(str(np.format_float_scientific(i, precision=2))))
        return '\n'.join(['[' + ' '.join([toeq(str(np.format_float_scientific(x, precision=2)), num) for x in A[i]]) + ']' for i in range(len(A))])
    else:
        for i in A.flatten():
            num = max(num, len(str(np.around(i, 2))))
        return '\n'.join(['[' + ' '.join([toeq(str(np.around(x, 2)), num) for x in A[i]]) + ']' for i in range(len(A))])

def join_mat(A, B, s):
    A = A.split('\n')
    B = B.split('\n') 
    k = len(A[0])
    n = len(A)
    C = list(map(lambda p: p[0] + "   " + p[1], zip(A, B)))
    C[n//2] = C[n//2][:k+1] + s + C[n//2][k+2:]
    return '\n'.join(C)


def gaus(A: np.ndarray, b: np.ndarray):
    A2, b2 = np.copy(A), np.copy(b)
    n = b.shape[0]
    last = n
    per = np.zeros((n, n))
    for i in range(n):
        per[i][i] = 1.0
    for i in range(n):
        print('-' * 20)
        if i == last:
            break
        P = np.zeros((n, n))
        M = np.zeros((n, n))
        pos, maxi = -1.0, -1.0
        for j in range(i, n):
            if abs(A[i][j]) > maxi:
                pos, maxi = j, abs(A[i][j])
        if maxi == 0.0:
            last -= 1
            for j in range(n):
                if j < i:
                    P[j][j] = 1.0
                elif j == n-1:
                    P[j][i] = 1
                else:
                    P[j][j+1] = 1
            As = np.matmul(P, A)
            bs = np.matmul(P, b)
            print("A = P * A")
            print(join_mat(print_mat(As), join_mat(print_mat(P), print_mat(A), '*'), '='))
            print("b = P * b")
            print(join_mat(print_mat(bs), join_mat(print_mat(P), print_mat(b), '*'), '='))
            A =  np.copy(As)
            b = np.copy(bs)
            continue
        for j in range(n):
            P[j][j] = 1.0
            M[j][j] = 1.0
        P[[i, pos]] = P[[pos, i]]
        M[i][i] = 1 / A[i][pos]
        for j in range(i+1, n):
            if A[j][pos] != 0.0:
                M[j][i] = -A[j][pos] / A[i][pos]
        As = np.matmul(np.matmul(M, A), P)
        print("A = M * A * P")
        print(join_mat(join_mat(join_mat(print_mat(As), print_mat(M), '='), print_mat(A), '*'), print_mat(P), '*'))
        A = np.copy(As)
        pers = np.matmul(per, P)
        print("per = per * P")
        print(join_mat(print_mat(pers), join_mat(print_mat(per), print_mat(P), '*'), '='))
        per = np.copy(pers)
        bs = np.matmul(M, b)
        print("b = M * b")
        print(join_mat(print_mat(bs), join_mat(print_mat(M), print_mat(b), '*'), '='))
        b = np.copy(bs)
    x = np.zeros((n, 1))
    for i in range(n-1, -1, -1):
        res = b[i][0]
        for j in range(i+1, n):
            res -= x[j][0] * A[i][j]
        x[i][0] = res
    print("A, b > x")
    print(join_mat(print_mat(A), join_mat(print_mat(b), print_mat(x), '>'), ','))
    x = np.matmul(per, x)
    print('-' * 20)
    print("x = per * b")
    print(join_mat(print_mat(x), join_mat(print_mat(per), print_mat(b), '*'), '='))
    print("A * x = b")
    print(join_mat(print_mat(A2), join_mat(print_mat(x), print_mat(b2), '='), '*'))


def power_iteration(A: np.ndarray, eps=10**-6, iter=100):
    n = A.shape[0]
    x = np.random.randn(n, 1)
    for i in range(iter):
        xs = A @ x / np.linalg.norm(A @ x)
        if np.linalg.norm(xs - x) < eps:
            return (A @ xs)[0] / xs[0]
        x = xs
    return None


def jacobi(A: np.ndarray, b: np.ndarray, x=None, iter=200, eps=10**-6):
    n = A.shape[0]
    if not x:
        x = np.zeros((n, 1))
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = A[i][i]
    A -= D
    Di = np.linalg.inv(D)
    for i in range(iter):
        print(f"iter {i}")
        print(f"x = {x.flatten()}")
        print("x = D^-1 * (A - D) * x + D^-1 * b")
        xs = np.matmul(np.matmul(Di, A), x) + np.matmul(Di, b)
        print(join_mat(print_mat(xs), join_mat(print_mat(Di), join_mat(print_mat(A), join_mat(print_mat(x), join_mat(print_mat(Di), print_mat(b), '*'), '+'), '*'), '*'), '='))
        if np.linalg.norm(x - xs) < eps:
            print(f"algorithm has converged to x = {xs.flatten()}")
            return
        x = xs
    print("iteration limit exceeded, algorithm diverges")


'''
A = np.array([[2, 3, -2, 3], [-2, -4, 5, 0], [7, -2, -3, 1], [0, -4, 4, -1]], dtype=np.float64)
b = np.array([[2], [-3], [9], [1]], dtype=np.float64)
#gaus(A, b)
'''


'''
m = np.array([[ 7, -3,  3], [-5,  7, -8], [-9, -4, -8]], dtype=np.float64)
c = np.array([[6], [-4], [2]], dtype=np.float64)

d = np.zeros((3, 3))
for i in range(3): d[i, i] = m[i, i]
di = np.linalg.inv(d)
print(power_iteration(di @ (m - d)))

#jacobi(m, c)
'''