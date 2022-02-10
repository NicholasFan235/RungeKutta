import numpy as np
import typing

class RKSolver:
    def __init__(self, step_size):
        self._step_size = step_size
        self.y, self.t = None, None
        self._func = None

    def set_func(self, func : typing.Callable):
        self._func = func
    
    def set_state(self, y : np.ndarray, t : float = 0) -> None:
        self.y = np.array(y, dtype=float).reshape(-1); self.t = t

    def set_step_size(self, step_size: float) -> None:
        self._step_size = step_size
    
    def step(self) -> typing.Tuple[np.ndarray, float]:
        raise NotImplementedError
    
    def solve_times(self, times: np.ndarray, func:typing.Callable=None, y0:np.ndarray=None, t0: float = None, step_size:float=None) -> np.ndarray:
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
        