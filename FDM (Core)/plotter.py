# plotter.py
import matplotlib.pyplot as plt

class PoissonPlotter:
    def __init__(self, X, Y, u_exact, u_numeric, error_matrix):
        self.X = X
        self.Y = Y
        self.u_exact = u_exact
        self.u_numeric = u_numeric
        self.error_matrix = error_matrix

    def plot_contours(self):
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.contourf(self.X, self.Y, self.u_exact, levels=50, cmap='jet')
        plt.colorbar()
        plt.title('Exact Solution (Contour)')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.contourf(self.X, self.Y, self.u_numeric, levels=50, cmap='jet')
        plt.colorbar()
        plt.title('Numerical Solution (Contour)')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.contourf(self.X, self.Y, self.error_matrix, levels=50, cmap='hot')
        plt.colorbar(label='Absolute Error')
        plt.title('Error Heatmap')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()
