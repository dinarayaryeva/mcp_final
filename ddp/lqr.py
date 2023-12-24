from time import time
import logging
import numpy as np
from numpy.linalg import pinv
import sympy as sym
from sympy import lambdify

from collections.abc import Callable
from numpy.typing import ArrayLike


class LQROPtimizer:
    """Finite horizon Linear Quadratic Regulator (LQR)"""

    def __init__(
        self,
        Nx: int,
        Nu: int,
        dynamics: Callable,
        inst_cost: Callable,
        terminal_cost: Callable
    ):
        """
        Instantiates a DDP Optimizer and pre-computes the dynamics
        and cost derivates without doing any optimization/solving.

        :param Nx: dimension of the state variable x
        :param Nu: dimension of the control variable u
        :param dynamics: a callable dynamics function with 3 arguments
            x, u, constrain. This function must be closed-form differentiable
            by sympy. In other words, it must be built with sympy and/or numpy.
            Has to return the next state x' with same dimensions as input state.
        :param inst_cost: instantenious (aka running) cost funciton. Must be a
            callable function with 3 arguments x, u, x_goal. Again, must be
            closed-form differentiatable by sympy.
        :param term_cost: terminal cost funciton. Must be a
            callable function with 3 arguments x, u, x_goal. Again, must be
            closed-form differentiatable by sympy.
        """

        self.Nx = Nx
        self.Nu = Nu

        # Pre-compute derivatives now so that we don't have to do it every time
        x = sym.symbols("x:{:}".format(Nx))
        x = sym.Matrix([xi for xi in x])
        u = sym.symbols("u:{:}".format(Nu))
        u = sym.Matrix([ui for ui in u])
        x_goal = sym.symbols("x_g:{:}".format(Nx))
        x_goal = sym.Matrix([xi for xi in x_goal])

        # dynamics
        self.f = lambdify((x, u), dynamics(x, u))
        self.fx = lambdify((x, u), dynamics(x, u).jacobian(x))
        self.fu = lambdify((x, u), dynamics(x, u).jacobian(u))

        # costs
        self.g = lambdify((x, u, x_goal), inst_cost(x, u, x_goal))
        self.gx = lambdify((x, u, x_goal), inst_cost(x, u, x_goal).jacobian(x))
        self.gu = lambdify((x, u, x_goal), inst_cost(x, u, x_goal).jacobian(u))
        self.gxx = lambdify(
            (x, u, x_goal), inst_cost(x, u, x_goal).jacobian(x).jacobian(x)
        )
        self.gux = lambdify(
            (x, u, x_goal), inst_cost(x, u, x_goal).jacobian(u).jacobian(x)
        )
        self.guu = lambdify(
            (x, u, x_goal), inst_cost(x, u, x_goal).jacobian(u).jacobian(u)
        )
        self.h = lambdify((x, x_goal), terminal_cost(x, x_goal))
        self.hx = lambdify((x, x_goal), terminal_cost(x, x_goal).jacobian(x))
        self.hxx = lambdify(
            (x, x_goal), terminal_cost(x, x_goal).jacobian(x).jacobian(x)
        )

    def optimize(
        self,
        x0: ArrayLike,
        x_goal: ArrayLike,
        N: int = None,
        U0: ArrayLike = None,
        full_output: bool = False,
    ):
        """
        Optimize a trajectory given a starting state and a goal state.
        Note that the lenght of the trajectory is decided based on the args.

        :param x0: starting state. Must be of dimensions (Nx,1)
        :param x_goal: goal state. Must be of dimensions (Nx,1)
        :param N: trajectory lenght. If provided, the optimizer generates
            a random initial control sequence. This is called "slow start"
            and often results in poor optimization time (>1s)
        :param U0: initial control sequence. Must be of dimensions (N,Nu)
            where the N is the implied trajectory lenght. If provided this
            will be used to "warm start" the optimization, resulting in
            faster convergence rates (if the warm start is good)
        :param full_output: By default this function returns only the
            optimal state and control sequences. If full_output=True
            it also returns (optimal state sequence, optimal control sequence,
            state sequence history, control sequence history, total cost history)
        """

        if not N and not U0:
            print(
                "ERROR: You have to provide either trajectory length N or initial control sequency U0"
            )
            return

        x0 = np.array(x0)
        x_goal = np.array(x_goal)

        # Defined total cost of trajectory function
        # Note: defined here to give the flexibility of parameteraising
        #   x_goal on the fly
        def J(X, U):
            total_cost = 0.0
            for i in range(len(U)):
                total_cost += self.g(X[i], U[i], x_goal)
            total_cost += self.h(X[-1], x_goal)
            return float(total_cost)

        X = np.zeros((N + 1, self.Nx))
        X[0] = x0
        U = np.zeros((N, self.Nu))
        for i in range(len(U)):
            X[i + 1] = self.f(X[i], U[i]).flatten()

        # Matrix to store cost-to-go
        P = np.zeros((N, self.Nx, self.Nx))
        P[N-1] = self.hxx(X[N-1], x_goal)
        # Matrix to store gains
        K = np.zeros((N, self.Nu, self.Nx))

        # backwards pass
        for t in reversed(range(N)):
            A = self.fx(X[t], U[t])
            B = self.fu(X[t], U[t])
            Q = self.gxx(X[t], U[t], x_goal)
            R = self.guu(X[t], U[t], x_goal)
            # Compute the LQR gain
            M_k = R + B.T @ P[t] @ B
            invM_k = pinv(M_k)
            K[t-1] = invM_k @ B.T @ P[t] @ A
            Ac_k = A - B @ K[t-1]   # Update with difference Riccati equation
            # Compute cost-to-go matrix
            P[t-1] = Q + K[t-1].T @ R @ K[t-1] + Ac_k @ P[t] @ Ac_k

        for t in (range(N)):
            U[t] = - np.dot(K[t], X[t]).flatten()
            X[t+1] = self.f(X[t], U[t]).flatten()
        
        # calculate the cost
        cost = J(X, U)

        # keep a history of the trajectory
        if full_output:
            X_hist = [X.copy()]
            U_hist = [U.copy()]
            cost_hist = [cost]

        if full_output:
            return X, U, X_hist, U_hist, cost_hist

        return X, U
