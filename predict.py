import numpy as np
from numba import cuda, float64, jit
from scipy import sparse
import time
import heapq
import csv
import pandas as pd


TPB=32
@cuda.jit
def fast_matmul(a, b, c):
    sa = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sb = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # bpg = cuda.gridDim.x

    if x >= c.shape[0] and y >= c.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    tmp = 0
    for i in range(hid):
        sa[tx, ty] = a[x, ty + i * TPB]
        sb[tx, ty] = b[tx + i * TPB, y]
        cuda.syncthreads()

        for j in range(TPB):
            tmp += sa[tx, j] * sb[j, ty]
        cuda.syncthreads()
    c[x, y] = tmp


m_ = 640
n_ = 9760
k = 128 # 超参数 factor长度
multiple = np.zeros([m_,n_])
hid = int((k-1)/TPB)+1
blockdim = (TPB, TPB)
griddim1 = (20, 305)

Q = np.load("Q.npy")
P = np.load("P.npy")


time_start=time.time()
Movie_predict = np.dot(Q, P)
time_end=time.time()
print("np.dot耗时%ss" %(time_end - time_start))


time_start=time.time()
fast_matmul[griddim1, blockdim](Q, P, multiple)
time_end=time.time()
print("CUDA耗时%ss" %(time_end - time_start))

'''
row = np.array(np.load("all_set.npy")[:,0], dtype='float32')
col = np.array(np.load("all_set.npy")[:,1], dtype='float32')
data = np.array(np.load("all_set.npy")[:,2], dtype='float32')
movie = sparse.coo_matrix((data,(row,col)),shape=(m_,n_)).tocsr()


print(movie[0,0:100])
print(Movie_predict[0,0])
print(Movie_predict[0,2])
print(Movie_predict[0,5])
print(Movie_predict[0,43])
print(Movie_predict[0,46])
print(Movie_predict[0,62])
print(Movie_predict[0,89])
print(Movie_predict[0,97])
'''

with open("movies.csv", encoding='utf-8') as csvMovie:
    reader = csv.reader(csvMovie)
    data = []
    for index, item in enumerate(reader):
        if index != 0:
            data.append(item)
    print(np.shape(data))
    a = list(Movie_predict[1, :])
    re2 = heapq.nlargest(100, a)  # value
    re1 = list(map(a.index, re2))  # index

    predict = []
    for i in range (100):
        predict.append(data[re1[i]])
    csvFile = open('2.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    for index,item in enumerate(predict):
        writer.writerow(item)
    csvFile.close()
    print(predict)
    print(re2)


# print(Movie_predict[0,0:30])

