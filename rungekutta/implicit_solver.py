import numpy as np
import typing
import rungekutta as rk
import scipy.linalg

class ImplicitSolver(rk.RKSolver):
    """Implicit Runge-Kutta Solver
    Implementation of a generic Implicit Runge-Kutta algorithm.
    The Butcher Tableau must be specified to configure the solver.
    Weights and Nodes must have the same length (N).
    The Runge-Kutta matrix should be an NxN matrix.

    This Solver can solve any equation of the form dy/dt = A(t)y.
    The matrix A(t) is the function which must be configured in this solver.

    :param weights: weight values in Butcher Tableau
    :type weights: array-like
    :param nodes: node values in Butcher Tablue
    :type nodes: array-like
    :param rk_matrix: Runge-Kutta matrix in Butcher Tableau
    :type rk_matrix: array-like
    """

    def __init__(self, weights: np.ndarray, nodes: np.ndarray, rk_matrix: np.ndarray, step_size:float=1e-5):
        """Constructor Method
        """
        self._weights = np.array(weights).reshape(-1)
        self._nodes = np.array(nodes).reshape(-1)
        self._rk_matrix = np.array(rk_matrix)
        self._stages = self._nodes.shape[0]
        assert self._weights.shape[0] == self._stages
        assert self._rk_matrix.shape[0] == self._stages
        assert self._rk_matrix.shape[1] == self._stages
        super().__init__(step_size)

    def step(self) -> typing.Tuple[np.ndarray, float]:
        """Perform a single integration step
        Performs an integration step with configured step_size
        
        :returns: Tuple of state and time after integration step
        :rtype: Tuple[np.ndarray, float]
        """
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
    