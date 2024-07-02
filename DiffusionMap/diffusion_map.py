import numpy as np
from scipy.sparse.linalg import svds
from scipy.spatial.distance import squareform, cdist, pdist

class DIFFUSION_MAP:
    def __init__(self, sigma, c=3, opt='global'):
        self.sigma = sigma
        self.c = c
        self.opt = opt

    def train(self, data):
        data = data.reshape(data.shape[0], -1)
        K = self.get_K(data)
        Q_hat, D_hat = self.get_QD(K)  # D_hat = D^(-1/2)
        self.eig_vals, eig_vecs = self.get_eig(Q_hat, D_hat, c=self.c)
        self.Y = eig_vecs.dot(np.diag(self.eig_vals))[:, 1:]
        return self

    def get_K(self, data):
        Dis = squareform(pdist(data, metric="euclidean"))
        Dis_sort = np.sort(squareform(pdist(data)), 1)
        if self.opt == 'global':
            # eps = 2 * np.percentile(Dis_sort, 25) ** 2
            eps = np.sqrt(np.median(Dis))
            K = np.exp(- Dis * Dis / eps)
        elif self.opt == 'local':
            Sigma = np.diag(1/Dis_sort[:, self.sigma + 1])
            K = Sigma.dot(Dis * Dis).dot(Sigma)
            K = np.exp(-K)
        return K

    def get_QD(self, K):
        D = np.diag(1/np.sqrt(np.sum(K, axis=1))) # row sum
        Q = D.dot(K).dot(D)
        return Q, D

    def get_eig(self, Q, D, c):
        np.random.seed(505)
        U, S, _ = svds(Q, c + 1)
        eig_vals = S[::-1]
        eig_vecs = D.dot(U[:, ::-1])
        return eig_vals, eig_vecs
    
class ROSELAND:
    def __init__(self, sigma, c=3, opt='global'):
        self.sigma = sigma
        self.c = c
        self.opt = opt

    def train(self, data, land):
        W_r = self.get_W(data, land)  # construct affinity matrix
        self.eig_vals, eig_vecs = self.get_eig(W_r, c=self.c)  # SVD

        eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)
        self.Y = eig_vecs.dot(np.diag(self.eig_vals))[:, 1:]
        return self

    def get_W(self, data, landmark):
        # data resize to 2D
        data = data.reshape(data.shape[0], -1)
        landmark = landmark.reshape(landmark.shape[0], -1)
        W = cdist(data, landmark)
        if (self.opt == 'global'):
            eps = np.sqrt(np.median(np.median(W, axis=1)))
            # eps =  2 * np.percentile(np.sort(W, 1), 25) ** 2
            W = np.exp(- W * W / eps)
            
        elif (self.opt == 'local'):
            tmp = np.sort(W, axis=0)[self.sigma + 1, :]
            tmp[np.argwhere(tmp == 0)] = 1e-8
            sigmasm = np.diag(1 / tmp)

            tmp = np.sort(W, axis=1)[:, self.sigma + 1]
            tmp[np.argwhere(tmp == 0)] = 1e-8
            sigmabg = np.diag(1 / tmp)

            W = np.exp(-sigmabg.dot(W * W).dot(sigmasm))
        return W

    def get_eig(self, W, c):
        s = np.sum(W.dot(W.T), axis=1)
        D = np.diag(1 / np.sqrt(s))
        Q = D.dot(W)
        
        np.random.seed(505)
        U, S, _ = svds(Q, c + 1)
        
        eigenvals = S[::-1] * S[::-1]
        eig_vecs = D.dot(U[:, ::-1])
        
        return eigenvals, eig_vecs