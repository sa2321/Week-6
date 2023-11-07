"""A module providing numerical solvers for nonlinear equations."""


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps, max_its):
    """Solve a nonlinear equation using Newton-Raphson iteration."""
    x = x_0
    for _ in range(max_its):
        x = x - f(x) / df(x)
        if abs(f(x)) < eps:
            return x

    raise ConvergenceError("Newton-Raphson iteration did not converge")
    raise NotImplementedError


def bisection(f, x_0, x_1, eps, max_its):
    """Solve a nonlinear equation using bisection."""
    if f(x_0) * f(x_1) >= 0:
        raise ValueError("Initial points must have opposite signs")

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
    """Solve nonlinear equation using Newton-Raphson and bisection."""
    try:
        return newton_raphson(f, df, x_0, eps, max_its_n)
    except ConvergenceError:
        return bisection(f, x_0, x_1, eps, max_its_b)
    raise NotImplementedError
