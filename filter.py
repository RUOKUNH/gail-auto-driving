import pdb

import numpy as np
import matplotlib.pyplot as plt


def kf_predict(X0, P0, A, Q, B, U1):
    X10 = np.dot(A,X0) + np.dot(B,U1)
    P10 = np.dot(np.dot(A,P0),A.T)+ Q
    return (X10, P10)

def kf_update(X10, P10, Z, H, R):
    V = Z - np.dot(H,X10)
    K = np.dot(np.dot(P10,H.T),np.linalg.pinv(np.dot(np.dot(H,P10),H.T) + R))
    X1 = X10 + np.dot(K,V)
    P1 = np.dot(1 - np.dot(K,H),P10)
    return (X1, P1, K)


def kalman_filter(x, v, a, dt):
    # x = v
    # v = a
    # a = a

    x0 = x[0]
    v0 = v[0]
    a0 = a[0]
    n = len(x)
    nx = 3

    R = np.diag([1])
    A = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
    B = 0
    U1 = 0
    X0 = np.array([x0,v0,a0]).reshape(nx,1)
    P0 = np.diag([0, 0.06**2, 0.06**2])
    Q = np.array([[0.001, 0.001, 0.0], [0.001, 0.001, 0.0], [0.0, 0.0, 0.0]])
    X1_np = np.copy(X0)
    P1_list = [P0]
    X10_np = np.copy(X0)
    P10_list = [P0]
    for i in range(n):
        Zi = np.array(x[i]).reshape([1, 1])
        Hi = np.array([1, 0, 0]).reshape([1, nx])

        if i == 0:
            continue
        else:
            Xi = X1_np[:, i - 1].reshape([nx, 1])
            Pi = P1_list[i - 1]
            X10, P10 = kf_predict(Xi, Pi, A, Q, B, U1)

            X10_np = np.concatenate([X10_np, X10], axis=1)
            P10_list.append(P10)

            X1, P1, K = kf_update(X10, P10, Zi, Hi, R)
            X1_np = np.concatenate([X1_np, X1], axis=1)
            P1_list.append(P1)

    xf, vf, af = X1_np.tolist()

    # plt.plot(range(len(x)), x)
    # plt.plot(range(len(x)), xf)
    # plt.savefig('test/xf.png')
    # plt.close()
    # # plt.show()
    # plt.plot(range(len(v)), v)
    # plt.plot(range(len(v)), vf)
    # plt.savefig('test/vf.png')
    # plt.close()
    # # plt.show()
    # plt.plot(range(len(a)), a)
    # plt.plot(range(len(a)), af)
    # plt.savefig('test/af.png')
    # plt.close()
    # plt.show()
    return xf, vf, af
