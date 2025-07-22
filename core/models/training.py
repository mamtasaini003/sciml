import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from core.physics.poissions2d import pde_loss_poissions2d 

class Training:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, pde_data, boundary_data):
        with tf.GradientTape() as tape:
            loss_pde = pde_loss_poissions2d.physics_loss(self.model, pde_data)
            loss_bc = pde_loss_poissions2d.boundary_loss(self.model, boundary_data)
            total_loss = loss_pde + 10 * loss_bc
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss, loss_pde, loss_bc

    def train(self, model, pde_data, boundary_data, epochs=10, log_epoch=100):
        total_loss_lst = []
        pde_loss_lst = []
        bc_loss_lst = []

        for epoch in range(epochs):
            total_loss, loss_pde, loss_bc = Training.train_step(self, pde_data, boundary_data)
            total_loss_lst.append(total_loss.numpy())
            pde_loss_lst.append(loss_pde.numpy())
            bc_loss_lst.append(loss_bc.numpy())

            if epoch % log_epoch == 0 or epoch == epochs-1:
                print(f"Epoch {epoch}: Total= {total_loss.numpy()}, PDE loss:={loss_pde.numpy()}, BC loss:={loss_bc.numpy()}")

                # # Plot loss curves
                # plt.figure()
                # plt.plot(total_loss_lst, label="Total Loss")
                # plt.plot(pde_loss_lst, label="PDE Loss")
                # plt.plot(bc_loss_lst, label="BC Loss")
                # plt.xlabel('Epoch')
                # plt.ylabel('Loss')
                # plt.yscale('log')
                # plt.legend()
                # plt.savefig("core/output/poissons2d/loss_component_plot.png")

                # def generate_test_points(xmin, xmax, ymin, ymax, N_test):
                #     x = np.linspace(xmin, xmax, N_test)
                #     y = np.linspace(ymin, ymax, N_test)
                #     X, Y = np.meshgrid(x, y)
                #     test_points = np.column_stack((X.flatten(), Y.flatten()))
                #     return tf.convert_to_tensor(test_points, dtype=tf.float32), X, Y

                # def exact_solution(points):
                #     x = points[:, 0]
                #     y = points[:, 1]
                #     return np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)

                # N_test = 100
                # test_points, X, Y = generate_test_points(0, 1, 0, 1, N_test)

                # u_pred = model(test_points)
                # u_pred = u_pred.numpy().flatten()
                # u_exact = exact_solution(test_points.numpy())

                # abs_error = np.abs(u_exact-u_pred)
                # abs_error_grid = abs_error.reshape(X.shape)
                # u_pred_grid = u_pred.reshape(X.shape)
                # u_exact_grid = u_exact.reshape(X.shape)

                # plt.figure(figsize=(20, 5))

                # # Exact solution
                # plt.subplot(1, 3, 1)
                # plt.contourf(X, Y, u_exact_grid, levels=100, cmap='jet')
                # plt.colorbar()
                # plt.title("Exact Solution")

                # # Predicted solution
                # plt.subplot(1, 3, 2)
                # plt.contourf(X, Y, u_pred_grid, levels=100, cmap='jet')
                # plt.colorbar()
                # plt.title("Predicted (PINN)")

                # # Absolute Error
                # plt.subplot(1, 3, 3)
                # plt.contourf(X, Y, abs_error_grid, levels=100, cmap='hot')
                # plt.colorbar()
                # plt.title("Absolute Error")

                # plt.tight_layout()
                # plt.savefig("core/output/poissons2d/result_u_pred.png")
        return total_loss_lst, pde_loss_lst, bc_loss_lst
    