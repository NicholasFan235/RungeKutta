import numpy as np
import typing

class RKSolver:
    """Abstract Parent for Runge-Kutta Solvers
    Provides implementation of functions common between Explicit and Implicit solvers

    :param step_size: Step size to use when integrating for solution
    :type step_size: float
    """

    def __init__(self, step_size):
        """Constructor Method
        
        :param step_size: Step size to integrate with
        :type step_size: float
        """
        self._step_size = step_size
        self.y, self.t = None, None
        self._func = None

    def set_func(self, func : typing.Callable):
        """Set the function to be integrated
        
        :param func: Function to be integrated
        :type func: Callable
        """
        self._func = func
    
    def set_state(self, y : np.ndarray, t : float = 0) -> None:
        """Set the initial conditions

        :param y: state of y at time t
        :type y: array-like
        :param t: time t
        :type t: float
        """
        self.y = np.array(y, dtype=float).reshape(-1); self.t = t

    def set_step_size(self, step_size: float) -> None:
        """Set the step size to be used

        :param step_size: step size for integrator to use
        :type step_size: float
        """
        self._step_size = step_size
    
    def step(self) -> typing.Tuple[np.ndarray, float]:
        """Abstract method to be implemented by the solvers
        Performs one integration step and returns the state

        :returns: Tuple of y and t. (state at time t)
        :rtype: Tuple[np.ndarray, float]
        """
        raise NotImplementedError
    
    def solve_times(self, times: np.ndarray, func:typing.Callable=None, y0:np.ndarray=None, t0: float = None, step_size:float=None) -> np.ndarray:
        """Integrate and return the state at the specified times
        Integrates the system using the step method provided by the child implementation.
        The configured step_size should be smaller than the interval between specified times

        :param times: Time points to return the states at
        :type times: array-like
        :param func: Function to integrate. Uses the configured function if none is specified
        :type func: Callable, optional
        :param y0: Initial y. Uses the configured state if none is specified
        :type y0: array-like
        :param t0: Initial t. Uses the configured time if none is specified
        :type t0: float, optional
        :param step_size: Step size to integrate with. Uses the configured step_size if none is specified
        :type step_size: float, optional
        :returns: array of states for the specified times
        :rtype: np.ndarray
        """
        if func is not None: self._func = func
        if step_size is not None: self._step_size = step_size
        if y0 is not None: self.y = np.array(y0, dtype=float).reshape(-1)
        if t0 is not None: self.t = t0
        
        times = np.array(times, dtype=float).reshape(-1)
        ys = np.zeros((times.shape[0], self.y.shape[0]))
        for i in range(times.shape[0]):
            while self.t < times[i]:
                self.step()
            ys[i, :] = self.y
        return ys
        