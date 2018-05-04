import numpy as np
import sklearn.feature_selection as sklfs
import scipy as sc
import cvxpy as cvx


def corr(X, Y=None, fill=0):
    if Y is None:
        Y = X
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
        Y = np.atleast_2d(Y)
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
        corr[idxs_nz_x[i], idxs_nz_y] = Y_.T.dot(x) / np.sqrt(np.sum(x ** 2) * np.sum(Y_ ** 2, axis=0, keepdims=True))
    return corr


def shift_spectrum(Q, eps=0.):
    lamb_min = sc.linalg.eigh(Q)[0][0]
    if lamb_min < 0:
        Q = Q - (lamb_min - eps) * np.eye(*Q.shape)
    return Q, lamb_min


class QPFS:
    def __init__(self, sim='corr'):
        if sim not in ['corr', 'info']:
            raise ValueError('Similarity measure should be "corr" or "info"')
        self.sim = sim
    
    def get_params(self, X, y):
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

    def fit(self, X, y):
        self.get_params(X, y)
        alpha = self.get_alpha()
        self.solve_problem(alpha)
    
    def solve_problem(self, alpha):
        n = self.Q.shape[0]
        x = cvx.Variable(n)
        c = np.ones((n, 1))
        objective = cvx.Minimize((1 - alpha) * cvx.quad_form(x, self.Q) - 
                                 alpha * self.b.T * x)
        constraints = [x >= 0, c.T * x == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve()

        self.status = prob.status
        self.score = np.array(x.value).flatten()
        
    def get_topk_indices(self, k=10):
        return self.score.argsort()[::-1][:k]


class MultivariateQPFS():
    def __init__(self):
        pass
    
    def get_params(self, X, Y, eps=1e-12):
        self.Qx = np.abs(corr(X, fill=1))
        self.Qy = np.abs(corr(Y, fill=1))
        np.fill_diagonal(self.Qx, 0)
        np.fill_diagonal(self.Qy, 0)
        self.B = np.abs(corr(X, Y))

    def get_alpha(self, alpha3=None):
        if alpha3 is None:
            alpha3 = np.mean(self.Qx) * np.mean(self.B) / (np.mean(self.Qx) * np.mean(self.B) + 
                                                           np.mean(self.Qx) * np.mean(self.Qy) + 
                                                           np.mean(self.Qy) * np.mean(self.B))
        alpha1 = (1 - alpha3) * np.mean(self.B) / (np.mean(self.Qx) + np.mean(self.B))
        alpha2 = (1 - alpha3) * np.mean(self.Qx) / (np.mean(self.Qx) + np.mean(self.B))
        return np.array([alpha1, alpha2, alpha3])


    def fit(self, X, Y):
        self.get_params(X, Y)
        alphas = self.get_alpha()
        self.solve_problem(alphas)
    
    def solve_problem(self, alphas):
        n = self.Qx.shape[0]
        r = self.Qy.shape[0]
        
        Q = np.vstack((np.hstack((alphas[0] * self.Qx, -alphas[1] / 2 * self.B)),
                       np.hstack(( -alphas[1] / 2 * self.B.T, alphas[2] * self.Qy))))
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
        
    def get_topk_indices(self, k=10):
        return self.score[self.B.shape[0]].argsort()[::-1][:k] 
    

class MinimaxQPFS():
    def __init__(self, mode='minmax'):
        self.mode = mode
    
    def get_params(self, X, Y):
        self.Qx = np.abs(corr(X, fill=1))
        self.Qy = np.abs(corr(Y, fill=1))
        self.B = np.abs(corr(X, Y))

    def get_alpha(self, alpha3=None):
        if alpha3 is None:
            alpha3 = np.mean(self.Qx) * np.mean(self.B) / (np.mean(self.Qx) * np.mean(self.B) + 
                                                           np.mean(self.Qx) * np.mean(self.Qy) + 
                                                           np.mean(self.Qy) * np.mean(self.B))
        alpha1 = (1 - alpha3) * np.mean(self.B) / (np.mean(self.Qx) + np.mean(self.B))
        alpha2 = (1 - alpha3) * np.mean(self.Qx) / (np.mean(self.Qx) + np.mean(self.B))
        return np.array([alpha1, alpha2, alpha3])


    def fit(self, X, Y):
        self.get_params(X, Y)
        alphas = self.get_alpha()
        self.solve_problem(alphas)
    
    def solve_problem(self, alphas):
        n = self.Qx.shape[0]
        r = self.Qy.shape[0]
        
        np.fill_diagonal(self.Qx, 0)
        np.fill_diagonal(self.Qy, 0)
        self.Qx, lamb_min = shift_spectrum(self.Qx, eps=1e-6)
        self.Qy, lamb_min = shift_spectrum(self.Qy, eps=1e-6)
        
        if self.mode == 'maxmin':
            self._maxmin(alphas)
        elif self.mode == 'minmax':
            self._minmax(alphas)
        elif self.mode == 'dual_woy':
            self._dual_woy(alphas)
        
    def _minmax(self, alphas):
        n = self.Qx.shape[0]
        r = self.Qy.shape[0]
        Qyinv = np.linalg.pinv(self.Qy)
        Q1 = alphas[1] ** 2 * self.B.dot(Qyinv).dot(self.B.T) + 4 * alphas[0] * alphas[2] * self.Qx
        Q2 = Qyinv
        Q3 = Qyinv.sum(keepdims=True)
        Q12 = alphas[1] * self.B.dot(Qyinv)
        Q13 = alphas[1] * Q12.sum(axis=1, keepdims=True)
        Q23 = Qyinv.sum(axis=1, keepdims=True)
        
        Q = np.vstack([
            np.hstack([Q1, -Q12, -Q13]),
            np.hstack([-Q12.T, Q2, Q23]),
            np.hstack([-Q13.T, Q23.T, Q3])
        ])
        
        Q, lamb_min = shift_spectrum(Q)
        
        a = cvx.Variable(n + r + 1)
        
        c = np.zeros(n + r + 1)
        c[:n] = 1
        
        objective = cvx.Minimize(cvx.quad_form(a, Q) - 4 * alphas[2] * a[-1])
        constraints = [a[:-1] >= 0, c * a == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve(solver=cvx.CVXOPT)

        score = np.array(a.value).flatten()
        self.lamb = score[-1]
        self.nu = score[n:-1]
        self.ax = score[:n]
        self.ay = 1. / (2 * alphas[2]) * Qyinv.dot(-alphas[1] * self.B.T.dot(self.ax) + score[-1] + score[n:-1])
        
        self.Q = Q
        self.score = score
    
    def _maxmin(self, alphas):
        n = self.Qx.shape[0]
        r = self.Qy.shape[0]
        
        Qxinv = np.linalg.pinv(self.Qx)
        Q1 = alphas[1] ** 2 * self.B.T.dot(Qxinv).dot(self.B) + 4 * alphas[0] * alphas[2] * self.Qy
        Q2 = Qxinv
        Q3 = Qxinv.sum(keepdims=True)
        Q12 = alphas[1] * self.B.T.dot(Qxinv)
        Q13 = alphas[1] * Q12.sum(axis=1, keepdims=True)
        Q23 = Qxinv.sum(axis=1, keepdims=True)
        
        Q = np.vstack([
            np.hstack([Q1, Q12, Q13]),
            np.hstack([Q12.T, Q2, Q23]),
            np.hstack([Q13.T, Q23.T, Q3])
        ])
        
        Q, lamb_min = shift_spectrum(Q)
        self.lamb_min = lamb_min
        
        a = cvx.Variable(n + r + 1)
        
        c = np.zeros(n + r + 1)
        c[:r] = 1
        
        objective = cvx.Minimize(cvx.quad_form(a, Q) - 4 * alphas[0] * a[-1])
        constraints = [a[:-1] >= 0, c * a == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve(solver=cvx.CVXOPT)
        
        self.status = prob.status
        score = np.array(a.value).flatten()
        self.lamb = score[-1]
        self.mu = score[r:-1]
        self.ay = score[:r]
        
        self.ax = 1. / (2 * alphas[0]) * Qxinv.dot(alphas[1] * self.B.dot(self.ay) + score[-1] + score[r:-1])
        
        a = cvx.Variable(n)
        
        c = np.ones(n)
        
        objective = cvx.Minimize(cvx.quad_form(a, self.Qx) - self.B.dot(self.ay).flatten() * a)
        constraints = [a >= 0, c * a == 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve(solver=cvx.CVXOPT)
        self.ax = np.array(a.value).flatten()
        
    def _sdp(self, alphas):
        n = self.Qx.shape[0]
        r = self.Qy.shape[0]
        
        Qxinv = np.linalg.pinv(self.Qx)
        Q1 = alphas[1] ** 2 * self.B.T.dot(Qxinv).dot(self.B) + 4 * alphas[0] * alphas[2] * self.Qy
        Q2 = Qxinv
        Q3 = Qxinv.sum(keepdims=True)
        Q12 = alphas[1] * self.B.T.dot(Qxinv)
        Q13 = alphas[1] * Q12.sum(axis=1, keepdims=True)
        Q23 = Qxinv.sum(axis=1, keepdims=True)
        
        Q = np.vstack([
            np.hstack([Q1, Q12, Q13]),
            np.hstack([Q12.T, Q2, Q23]),
            np.hstack([Q13.T, Q23.T, Q3])
        ])
        
        q = np.zeros(n + r + 1)
        q[-1] = -2. * alphas[0]
        Qt = np.zeros((n + r + 2, n + r + 2))
        Qt[:n + r + 1, :n + r + 1] = Q
        Qt[-1, :-1] = q
        Qt[:-1, -1] = q
        A = np.zeros((1, n + r + 2))
        A[0, :n] = 1
        b = 1
        
        Z = cvx.Semidef(n + r + 2)
        
        objective = cvx.Minimize(cvx.trace(Qt * Z))
        constraints = [cvx.trace(A.T.dot(A) * Z) == b ** 2]
        prob = cvx.Problem(objective, constraints)

        prob.solve(solver=cvx.CVXOPT)
        
        self.status = prob.status
    
    def _dual_woy(self, alphas):
        n = self.Qx.shape[0]
        r = self.Qy.shape[0]
        
        a = cvx.Variable(n + r + 1)
        C = np.zeros((r + 1, n + r + 1))
        C[0, :n] = 1
        C[1:, :n] = alphas[1] * self.B.T
        C[1:, n:n + r] = -np.eye(r)
        C[1:, n + r] = -1
        d = np.zeros(r + 1)
        d[0] = 1
        Q = np.zeros((n + r + 1, n + r + 1))
        Q[:n, :n] = self.Qx
        
        objective = cvx.Minimize(alphas[0] * cvx.quad_form(a, Q) - a[-1])
        constraints = [a[:-1] >= 0, C * a == d]
        prob = cvx.Problem(objective, constraints)

        prob.solve(solver=cvx.CVXOPT)
        self.status = prob.status
        score = np.array(a.value).flatten()
        self.lamb = score[-1]
        self.ax = score[:n]
    
    def get_topk_indices(self, k=10):
        return self.score[self.B.shape[0]].argsort()[::-1][:k]
    

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    qpfs = QPFS()
    qpfs.fit(X, y)
    print(qpfs.score)