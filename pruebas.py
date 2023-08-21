import numpy as np
import matplotlib.pyplot as plt

def position_encoding(d, n):    
    vals_i = np.arange(0, d)
    vals_i = np.tile(vals_i, (n, 1))
    pos = np.arange(0, n)
    pos = np.transpose(np.tile(pos, (d, 1)))        
    sins  = np.sin(pos / np.power(10000, 2*vals_i / d))
    cosins  = np.cos(pos / np.power(10000, 2*vals_i / d))
    pe =  np.where(vals_i % 2 == 0, sins, cosins) 
    return pe
    
def  compute_entropy(p,q) :
    H = p * np.log2(1.0 / q)
    return np.sum(H)

if __name__ == '__main__' :
    pe = position_encoding(512, 128)
    plt.imshow(pe, cmap = 'jet')
    plt.xlabel('Size of embedding')
    plt.ylabel('n positions (n tokens)')
    plt.show()
#     p = np.array([1/3, 1/3, 1/3])
#     q = np.array([3/5, 2/5])
#     h1 = compute_entropy(p, p)
#     h2 = compute_entropy(q, q)
#     print(h1)
#     print(h2)

    
    
