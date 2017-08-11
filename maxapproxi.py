import numpy as np

def softmax(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    p = e_x/e_x.sum()
    p = p/p.sum()

    return p

def logsumexp(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    lse_x = max_x + np.log(np.exp(x-max_x).sum())
    lse_x = scale*lse_x
    return lse_x

def sparsetau(x):
    x = np.array(x)
    sorted_x = np.sort(x)[::-1]
    S = np.array([])
    for i in range(0,len(x)):
        if 1+(i+1)*sorted_x[i]>=(sorted_x[0:(i+1)]).sum():
            S = np.append(S,sorted_x[i])
    tau = (S.sum() - 1)/S.size
    return tau, S

def sparsedist(x, scale = 1):
    x = np.array(x/scale)
    tau, _ = sparsetau(x)
    p = x - tau
    p[p<0] = 0
    if p.sum() > 0.0:
        p = p/p.sum()
    else:
        p = np.ones_like(x)/x.shape[0];
    return p

def sparsemax(x,scale = 1):
    x = np.array(x/scale)
    tau, S = sparsetau(x)
    spmax_x = 0.5*(S**2 - tau**2).sum() + 0.5
    spmax_x = scale*spmax_x
    return spmax_x

if __name__ == '__main__':
    print("Main Started")
    x = np.random.rand(5)
    print(x[range(0,len(x))])
    print(np.sort(x))
    print(np.max(x))
    print(logsumexp(x))
    print(sparsemax(x))
    print(softmax(x))
    print(sparsedist(x))