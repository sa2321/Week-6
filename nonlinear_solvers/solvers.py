"""A module providing numerical solvers for nonlinear equations."""


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """"""
    x = x_0
     for _ in range(max_its):
        x = x - f(x) / df(x)
        if abs(f(x)) < eps:
            return x
  
    raise ConvergenceError("Newton-Raphson iteration did not converge")
    raise NotImplementedError

def bisection(f, x_0, x_1, eps=1.0e-5, max_its=20):
    """"""
    if f(x_0) * f(x_1) >= 0:
        raise ValueError("Initial points must have opposite signs for bisection")

    for _ in range(max_its):
        x_star = (x_0 + x_1) / 2
        if abs(f(x_star)) < eps:
            return x_star
        if f(x_0) * f(x_star) > 0:
            x_0 = x_star
        else:
            x_1 = x_star

    raise ConvergenceError("Bisection method did not converge")
    raise NotImplementedError

def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the bisection
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """
    # Delete these two lines when implementing the method.
    raise NotImplementedError
