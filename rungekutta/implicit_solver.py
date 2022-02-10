import numpy as np
import typing
import rungekutta as rk
import scipy.linalg

class ImplicitSolver(rk.RKSolver):
    def __init__(self, weights: np.ndarray, nodes: np.ndarray, rk_matrix: np.ndarray, step_size:float=1e-5):
        self._weights = np.array(weights).reshape(-1)
        self._nodes = np.array(nodes).reshape(-1)
        self._rk_matrix = np.array(rk_matrix)
        self._stages = self._nodes.shape[0]
        assert self._weights.shape[0] == self._stages
        assert self._rk_matrix.shape[0] == self._stages
        assert self._rk_matrix.shape[1] == self._stages
        super().__init__(step_size)

    def step(self) -> typing.Tuple[np.ndarray, float]:
        A_k = np.zeros((self._stages, self._stages, self.y.shape[0]), dtype=float)
        b_k = np.zeros((self._stages, 1, self.y.shape[0]), dtype=float)
        for i in range(self._stages):
            A = self._func(self.t + self._weights[i] * self._step_size)
            for j in range(self._stages):
                A_k[i,j, :] = self._step_size * \
                    np.dot(A, np.repeat(self._rk_matrix[i,j], self.y.shape[0]))
            A_k[i,i, :] -= 1
            b_k[i,0,:] = -np.dot(A, self.y)
        k = np.zeros((self._stages, self.y.shape[0]), dtype=float)
        for n in range(self.y.shape[0]):
            k[:, n] = scipy.linalg.solve(A_k[:, :, n], b_k[:, :, n]).reshape(-1)
        self.y += self._step_size * np.dot(self._nodes, k)
        self.t += self._step_size
        return self.y, self.t
    