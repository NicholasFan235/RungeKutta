import numpy as np
import typing
import rungekutta as rk

class ExplicitSolver(rk.RKSolver):
    """Explicit Runge-Kutta Solver
    Implementation of a generic Explicit Runge-Kutta algorithm.
    The Butcher Tableau must be specified to configure the solver.
    Weights and Nodes must have the same length (N).
    The Runge-Kutta matrix should be an NxN, lower triangular matrix.

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
        self._weights = np.array(weights, dtype=float).reshape(-1)
        self._nodes = np.array(nodes, dtype=float).reshape(-1)
        self._rk_matrix = np.array(rk_matrix, dtype=float)
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
        k = np.zeros((self._stages, self.y.shape[0]), dtype=float)
        for s in range(self._stages):
            k[s, :] = np.array(self._func(
                self.y + self._step_size * np.dot(self._rk_matrix[s, :s], k[:s, :]),
                self.t + self._step_size * self._weights[s])).reshape(-1)
        self.y += self._step_size * np.dot(self._nodes, k)
        self.t += self._step_size
        return self.y, self.t
