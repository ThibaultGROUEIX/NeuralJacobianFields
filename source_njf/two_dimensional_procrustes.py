import numpy as np
from scipy.linalg import lstsq
import torch
def two_dimensional_procrustes(S:torch.Tensor,T:torch.Tensor):
    S = S.numpy()
    T = T.numpy()
    for i in range(S.shape[0]):
        s = S[i, :,:2]
        t = T[i, :,:2]
        S[i,:,:2] = two_dimensional_procrustes_numpy(s,t)
    return torch.from_numpy(S)
def two_dimensional_procrustes_numpy(S,T):
    '''
    find rigid trans f(z) == z*M+d s.t. S*M + d = T in least squares
    :param S: Source poiunt cloud
    :param T: Target point cloud
    :return: the aligned S*M + d
    '''
    #
    # M = [a,b;-b,a]
    # S = S_x,S_y
    # T_x = a*S_x -b* S_y + d_x
    # T_y = b*S_x +a* S_y + d_y

    #  [S_x -S_y 1 0;S_y S_x 0 1] *[a b d_x d_y] = T
    S_1 = S[:,:2].copy() #S * a b
    S_1[:,1] = -S_1[:,1] #S * a -b
    S_2 = S[:,[1,0]]

    R_1 = np.hstack([S_1,np.ones((S_1.shape[0],1)),np.zeros((S_1.shape[0],1))])
    R_2 = np.hstack([S_2, np.zeros((S_2.shape[0],1)), np.ones((S_2.shape[0], 1))])

    R = np.vstack((R_1,R_2))
    assert R.shape[1] == 4
    rhs = np.hstack([T[:,0],T[:,1]])
    res = lstsq(R, rhs)[0]
    a = res[0]
    b = res[1]
    d_x = res[2]
    d_y = res[3]

    M =  np.array([[a,b],[-b,a]])
    M = M / np.linalg.norm(M)
    d = np.array([d_x,d_y])
    return np.matmul(S,M)+d#,M,d

if __name__ == '__main__':
    S = np.random.random([20,2])
    S = np.hstack([S,np.zeros((S.shape[0],1))])
    T = S + 5
    Sp = two_dimensional_procrustes_numpy(S,T)
    print(Sp-T) #should be numerical zeros