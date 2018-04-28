import numpy as np
import sklearn.feature_selection as sklfs
import scipy as sc
import cvxpy as cvx


def corr(X, Y=None, fill=0):
    if Y is None:
        Y = X
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    X_ = (X - X.mean(axis=0))
    Y_ = (Y - Y.mean(axis=0))
    
    idxs_nz_x = np.where(np.sum(X_ ** 2, axis = 0) != 0)[0]
    idxs_nz_y = np.where(np.sum(Y_ ** 2, axis = 0) != 0)[0]
    
    X_ = X_[:, idxs_nz_x]
    Y_ = Y_[:, idxs_nz_y]
    
    corr = np.ones((X.shape[1], Y.shape[1])) * fill
    
    for i, x in enumerate(X_.T):
        corr[idxs_nz_x[i], idxs_nz_y] = Y_.T.dot(x) / np.sqrt(np.sum(x ** 2) / np.sum(Y_ ** 2, axis=0, keepdims=True))
    return corr


def shift_spectrum(Q):
    lamb_min = sc.linalg.eigh(Q)[0][0]
    if lamb_min < 0:
        Q = Q - lamb_min * np.eye(*Q.shape)
    return Q, lamb_min


class QPFS:
    def __init__(self, sim='corr'):
        if sim not in ['corr', 'info']:
            raise ValueError('Similarity measure should be "corr" or "info"')
        self.sim = sim
    
    def get_Qb(self, X, y, eps=1e-12):
        if self.sim == 'corr':
            self.Q = np.abs(corr(X, fill=1))
            self.b = np.sum(np.abs(corr(X, y)), axis=1)[:, np.newaxis]
        elif self.sim == 'info':
            self.Q = np.ones([X.shape[1], X.shape[1]])
            self.b = np.zeros((X.shape[1], 1))
            for j in range(n_features):
                self.Q[:, j] = sklfs.mutual_info_regression(X, X[:, j])
            if len(y.shape) == 1:
                self.b = sklfs.mutual_info_regression(X, y)[:, np.newaxis]
            else:
                for y_ in y:
                    self.b += sklfs.mutual_info_regression(X, y_)
        self.Q, self.lamb_min = shift_spectrum(self.Q)
    
    def get_alpha(self):
        return self.Q.mean() / (self.Q.mean() + self.b.mean())

    def fit(self, X, y, alpha=None):
        self.get_Qb(X, y)
        self.alpha = alpha if alpha else self.get_alpha()
        self.solve_problem()
    
    def solve_problem(self):
        n = self.Q.shape[0]
        x = cvx.Variable(n)
        c = np.ones((n, 1))
        objective = cvx.Minimize((1 - self.alpha) * cvx.quad_form(x, self.Q) - 
                                 self.alpha * self.b.T * x)
        constraints = [x >= 0, c.T * x == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve()

        self.status = prob.status
        self.score = np.array(x.value).flatten()
        
    def __repr__(self):
        return f'QPFS(sim="{self.sim}")'


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    qpfs = QPFS()
    qpfs.fit(X, y)
    print(qpfs.score)


class MultivariateQPFS:
    def __init__(self):
        pass
    
    def get_params(self, X, Y, eps=1e-12):
        self.Q_x = np.abs(corr(X, fill=1))
        self.Q_y = np.abs(corr(Y, fill=1))
        self.B = np.abs(corr(X, Y))

    def fit(self, X, Y):
        self.get_params(X, Y)
        alpha = np.mean(self.Q_x) / (np.mean(self.Q_x) + np.mean(self.B))
        self.solve_problem([1 - alpha, alpha, 0.])
    
    def solve_problem(self, alphas):
        n = self.Q_x.shape[0]
        r = self.Q_y.shape[0]
        
        Q = np.vstack((np.hstack((alphas[0] * self.Q_x, -alphas[1] / 2 * self.B)),
                       np.hstack(( -alphas[1] / 2 * self.B.T, alphas[2] * self.Q_y))))
        Q, lamb_min = shift_spectrum(Q)
        
        a = cvx.Variable(n + r)
        
        c = np.zeros((2, n + r))
        c[0, :n] = 1
        c[1, n:] = 1
        
        objective = cvx.Minimize(cvx.quad_form(a, Q))
        constraints = [a >= 0, c * a == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve()

        self.status = prob.status
        self.score = np.array(a.value).flatten()
        
    def __repr__(self):
        return f'QPFS(sim="{self.sim}")'