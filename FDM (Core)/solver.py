# solver.py
import numpy as np

class PoissonSolver:
    def __init__(self, N=20, L=1.0, omega=1.8, tol=1e-6, max_iter=10000):
        self.N = N
        self.L = L
        self.omega = omega
        self.tol = tol
        self.max_iter = max_iter
        self.h = L / (N + 1)

        self.x = np.linspace(0, L, N + 2)
        self.y = np.linspace(0, L, N + 2)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.f = self._rhs_function(self.X, self.Y)
        self.u = np.zeros_like(self.f)
        self.u_exact = self._exact_solution(self.X, self.Y)
        self.error_matrix = None

    def _rhs_function(self, X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    def _exact_solution(self, X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    def solve(self):
        N, h, omega = self.N, self.h, self.omega
        u = self.u
        f = self.f

        for it in range(self.max_iter):
            max_error = 0.0
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    u_new = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h**2 * f[i, j])
                    diff = u_new - u[i, j]
                    u[i, j] += omega * diff
                    max_error = max(max_error, abs(diff))
            if max_error < self.tol:
                print(f"Converged in {it} iterations.")
                break

        self.u = u
        self.error_matrix = np.abs(u - self.u_exact)
        print(f"Max absolute error: {np.max(self.error_matrix):.2e}")
