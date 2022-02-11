import rungekutta as rk
import numpy as np

def forward_euler()->rk.ExplicitSolver:
    """Create a Forward Euler Solver

    :returns: Explicit Solver configured with Forward Euler Butcher Tableau
    :rtype: :class:`rk.ExplicitSolver`
    """
    solver = rk.ExplicitSolver([0], [1], [[0]])
    return solver

def explicit_midpoint()->rk.ExplicitSolver:
    """Create an Explicit Midpoint Solver
    
    :returns: Explicit Solver with Butcher Table configured for 2nd order midpoint method with 2 stages
    :rtype: :class:`rk.ExplicitSolver`
    """
    solver = rk.ExplicitSolver([0, 0.5], [0, 1], [[0, 0], [1/2, 0]])
    return solver

def fourth_order_runge_kutta()->rk.ExplicitSolver:
    """Create a Fourth-Order Runge-Kutta solver
    The original Runge-Kutta method
    
    :returns: Explicit Solver with Butcher Tableau from the classic fourth-order Runge-Kutta matrix
    :rtype: :class:`rk.ExplicitSolver`
    """
    solver = rk.ExplicitSolver(
        [0, 0.5, 0.5, 1], [1/6, 1/3, 1/3, 1/6],
        [[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]])
    return solver

def dormand_prince()->rk.ExplicitSolver:
    """Create a Dormand-Prince Solver
    
    :returns: Explicit Solver with Dormand-Prince Butcher Tableau
    :rtype: :class:`rk.ExplicitSolver`
    """
    solver = rk.ExplicitSolver(
        [0,1/5,3/10,4/5,8/9,1,1], [35/384,0,500/1113,125/192,-2187/6784,11/84,0],
        [[0,0,0,0,0,0,0],[1/5,0,0,0,0,0,0],[3/40,9/40,0,0,0,0,0],[44/45,-56/15,32/9,0,0,0,0],
        [19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],
        [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
        [35/384,0,500/1113,125/192,-2187/6784,11/84,0]])
    return solver

def backward_euler()->rk.ImplicitSolver:
    """Creat a simple Backwards Euler Solver
    
    :returns: Simple Implicit Solver for Backward Euler
    :rtype: :class:`rk.ImplicitSolver`
    """
    solver = rk.ImplicitSolver([1], [1], [[1]])
    return solver

def implicit_midpoint()->rk.ImplicitSolver:
    """Create an Implicit Midpoint Solver
    
    :returns: Implicit solver with Midpoint Butcher Tableau
    :rtype: :class:`rk.ImplicitSolver`
    """
    solver = rk.ImplicitSolver([0.5], [1], [[0.5]])
    return solver

def crank_nicolson()->rk.ImplicitSolver:
    """Create a Crank-Nicolson Implicit Solver
    
    :returns: Implicit Solver configured with Crank-Nicolson Butcher Tableau
    :rtype: :class:`rk.ImplicitSolver`
    """
    solver = rk.ImplicitSolver([0,1], [0.5,0.5],[[0,0],[0.5,0.5]])
    return solver

def gauss_legendre_order_four()->rk.ImplicitSolver:
    """Create a Fourth Order Gauss Legendre Solver

    :returns: Implicit Solver with Fourth Order Gauss Legendre Butcher Tableau
    :rtype: :class:`rk.ImplicitSolver`
    """
    solver = rk.ImplicitSolver(
        [1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6],[0.5,0.5],
        [[1/4, 1/4-np.sqrt(3)/6],
        [1/4+np.sqrt(3)/6, 1/4]])
    return solver

def gauss_legendre_order_six()->rk.ImplicitSolver:
    """Create a Sixth Order Gauss Legendre Solver
    
    :returns: Implicit Solver with Sixth Order Gauss Legendre Butcher Tableau
    :rtype: :class:`rk.ImplicitSolver`"""
    solver = rk.ImplicitSolver(
        [1/2-np.sqrt(15)/10, 0.5, 1/2+np.sqrt(15)/10],
        [5/18, 4/9, 5/18],
        [[5/36, 2/9-np.sqrt(15)/15, 5/36-np.sqrt(15)/30],
        [5/36+np.sqrt(15)/24, 2/9, 5/36-np.sqrt(15)/24],
        [5/36+np.sqrt(15)/30, 2/9+np.sqrt(15)/15, 5/36]])
    return solver
