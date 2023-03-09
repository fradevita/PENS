import numpy as np

def ThomasSolver(a, b, c, d):
    
    assert(len(a) == len(b) == len(c) == len(d)), f"Arrays a, b, c and d must have same size."
    
    N = np.size(a)

    c1 = np.zeros_like(d)
    c1[0] = c[0]/b[0]
    d1 = np.zeros_like(d)
    d1[0] = d[0]/b[0]
    for i in range(1,N-1):
        c1[i] = c[i]/(b[i] - a[i]*c1[i-1])
        d1[i] = (d[i] - a[i]*d1[i-1])/(b[i] - a[i]*c1[i-1])

    if ((b[-1] - a[-1]*c1[-2]) == 0.):
        d1[-1] = 0.
    else:
        d1[-1] = (d[-1] - a[-1]*d1[-2])/(b[-1] - a[-1]*c1[-2])

    x = np.zeros_like(d1)
    x[-1] = d1[-1]
    for i in range(N-2,-1,-1):
        x[i] = d1[i] - c1[i]*x[i+1]

    return x