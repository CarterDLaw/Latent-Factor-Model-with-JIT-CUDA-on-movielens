import numpy as np
from numba import cuda, float64, jit
from scipy import sparse
import os
import time
time_start=time.time()
cuda.select_device(0)


# 使用jit模式加速单一元素乘法运算
@jit(nopython=True)
def multiply(c, d):
    return c * d


# 使用jit模式加速单一元素减法运算
@jit(nopython=True)
def subtract(e, f):
    return e - f


# 使用jit模式加速梯度下降算法
@jit(nopython=True)
def gradient_descent(err, p, la, lr, q):
    return q + ((err * p) - (la * q)) * lr


# Thread Per Block
TPB = int(32)


# 使用cuda.jit模式，用gpu加速矩阵的乘法运算
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


# Latent Factor Model的随机梯度下降算法
@jit(nopython=True)
def stochastic_loop(q_, p_, i_, j_, err_, la_, lr_, k_):
    for r_ in range(k_):
        q_[i_, r_] = gradient_descent(err_, p_[r_, j_], la_, lr_, q_[i_, r_])
        p_[r_, j_] = gradient_descent(err_, q_[i_, r_], la_, lr_, p_[r_, j_])


# 超参数的设置
iteration = 1000
learning_rate = 0.0002
lambda_ = 0.01
k = 128 # 超参数 factor长度
m = 610
m_ = 640
n = 9742
n_ = 9760


# 读训练数据集，存入稀疏矩阵movie_train中。
row_train = np.array(np.load("train_set.npy")[:,0], dtype='float32')
col_train = np.array(np.load("train_set.npy")[:,1], dtype='float32')
data_train = np.array(np.load("train_set.npy")[:,2], dtype='float32')
movie_train = sparse.coo_matrix((data_train,(row_train,col_train)),shape=(m_,n_)).tocsr()


# 读测试数据集，存入稀疏矩阵movie_test中。
row_test = np.array(np.load("test_set.npy")[:,0], dtype='float32')
col_test= np.array(np.load("test_set.npy")[:,1], dtype='float32')
data_test = np.array(np.load("test_set.npy")[:,2], dtype='float32')
movie_test = sparse.coo_matrix((data_test,(row_test,col_test)),shape=(m_,n_)).tocsr()


# 由于numba中，gpu的加速必须保证TPB * BPG恰好为所求矩阵的二维大小，因此要用a、b扩充矩阵大小
Q = np.random.rand(m,k) * 0.33
P = np.random.rand(k,n) * 0.33
a = np.zeros((30,k))  # 640 - 610 = 30
b = np.zeros((k,18))

'''
# Q与P为原参数矩阵，EXTEND为它们padding后的矩阵，对原矩阵运算没有影响
Q_EXTEND = np.row_stack((Q,a))  # (640,128)
P_EXTEND = np.column_stack((P,b))  # (128,9760)
'''

multiple = np.zeros([m_,n_])
hid = int((k-1)/TPB)+1
blockdim = (TPB, TPB)
griddim1 = (20, 305)

# 恢复上一次训练的参数
Q_EXTEND = np.load("Q.npy")
P_EXTEND = np.load("P.npy")

noise_q = np.random.rand(m_,k) * 0.003
noise_p = np.random.rand(k, n_) * 0.003
Q_EXTEND += noise_q - np.mean(noise_q)
P_EXTEND += noise_p - np.mean(noise_p)

compare_train = 0.05
compare_test = 0.10
flag = 0  # 学习速率自适应算法的参数

for t in range(iteration):
    loss_train = 0
    loss_test = 0

    # 确保进入cuda.jit的数据在内存中是连续的
    A = np.ascontiguousarray(Q_EXTEND)
    B = np.ascontiguousarray(P_EXTEND)
    fast_matmul[griddim1, blockdim](A, B, multiple)

    for index, item in enumerate(row_train):
        i = int(item)
        j = int(col_train[index])
        error = movie_train[i, j] - multiple[i, j]
        if index % 10000 == 0:
            print(error)
        loss_train += np.abs(error)

        # 对Q与P的extend矩阵中每个元素进行随机梯度下降
        stochastic_loop(Q_EXTEND, P_EXTEND, i, j, error, lambda_, learning_rate, k)
    loss_train /= 100685

    for index, item in enumerate(row_test):
        i = int(item)
        j = int(col_test[index])
        loss_test += np.abs(movie_test[i, j] - multiple[i, j])
    loss_test /= 151
    print('iteration%s, loss_train:%s, loss_test:%s' % (t, loss_train, loss_test))

    # 若几次迭代后，学习情况良好，则适当调高学习速率learning_rate，加快学习速度
    # 及时保存参数矩阵Q与P的EXTEND版本，以便分阶段进行优化
    if compare_train > loss_train and compare_test > loss_test:
        print("learning_rate:%s" % learning_rate)
        flag = flag + 1
        if flag == 2 and learning_rate < 0.0005:
            learning_rate = learning_rate + 0.00001
            flag = 0
        compare_train = loss_train
        compare_test = loss_test
        np.save("Q.npy", Q_EXTEND)
        np.save("P.npy", P_EXTEND)

    # 若本次迭代使得loss_train与loss_test中一者升高，则必须降低学习速率，避免梯度过大
    else:
        print("Warning! The loss is accumulating!")
        flag = 0
        learning_rate = learning_rate * 0.5
cuda.close()