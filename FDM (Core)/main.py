# main.py
from solver import PoissonSolver
from plotter import PoissonPlotter

def main():
    solver = PoissonSolver(N=20)
    solver.solve()

    plotter = PoissonPlotter(
        solver.X, solver.Y, solver.u_exact, solver.u, solver.error_matrix
    )
    plotter.plot_contours()

if __name__ == "__main__":
    main()
