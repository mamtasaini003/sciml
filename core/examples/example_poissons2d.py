import tensorflow as tf
import numpy as np
from core.models.training import Training
from core.models.pinn import PINN
import matplotlib.pyplot as plt
# from physics import poissions2d

# Hyperparameters
layers = [2, 30, 30, 30, 1]
lr = 0.001
epochs = 5000

# Prepare data
def get_boundary_pts(x_min, x_max, y_min, y_max, num_pts):
    pts_per_edge = num_pts // 4
    bottom = np.column_stack((np.linspace(x_min, x_max, pts_per_edge), np.full(pts_per_edge, y_min)))
    top = np.column_stack((np.linspace(x_min, x_max, pts_per_edge), np.full(pts_per_edge, y_max)))
    right = np.column_stack((np.full(pts_per_edge, x_max), np.linspace(y_min, y_max, pts_per_edge)))
    left = np.column_stack((np.full(pts_per_edge, x_min), np.linspace(y_min, y_max, pts_per_edge)))
    return np.vstack((bottom, top, right, left))

def get_interior_pts(x_min, x_max, y_min, y_max, num_pts):
    x = np.random.uniform(x_min, x_max, (num_pts, 1))
    y = np.random.uniform(y_min, y_max, (num_pts, 1))
    return np.hstack((x, y))

xmin, xmax, ymin, ymax = 0, 1, 0, 1
X_int = tf.convert_to_tensor(get_interior_pts(xmin, xmax, ymin, ymax, 1000), dtype=tf.float32)
X_b = tf.convert_to_tensor(get_boundary_pts(xmin, xmax, ymin, ymax, 1000), dtype=tf.float32)

# Model and training
model = PINN(layers)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
solver = Training(model, optimizer)
total_loss_list, pde_loss_list, bc_loss_list = solver.train(X_int, X_b, epochs)

solver.train(model, X_int, X_b, epochs)
# Plot loss curves
plt.figure()
plt.plot(total_loss_list, label="Total Loss")
plt.plot(pde_loss_list, label="PDE Loss")
plt.plot(bc_loss_list, label="BC Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig("core/output/poissons2d/loss_component_plot.png")

def generate_test_points(xmin, xmax, ymin, ymax, N_test):
    x = np.linspace(xmin, xmax, N_test)
    y = np.linspace(ymin, ymax, N_test)
    X, Y = np.meshgrid(x, y)
    test_points = np.column_stack((X.flatten(), Y.flatten()))
    return tf.convert_to_tensor(test_points, dtype=tf.float32), X, Y

def exact_solution(points):
    x = points[:, 0]
    y = points[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)

N_test = 100
test_points, X, Y = generate_test_points(0, 1, 0, 1, N_test)

u_pred = model(test_points)
u_pred = u_pred.numpy().flatten()
u_exact = exact_solution(test_points.numpy())

abs_error = np.abs(u_exact-u_pred)
abs_error_grid = abs_error.reshape(X.shape)
u_pred_grid = u_pred.reshape(X.shape)
u_exact_grid = u_exact.reshape(X.shape)

plt.figure(figsize=(20, 5))

# Exact solution
plt.subplot(1, 3, 1)
plt.contourf(X, Y, u_exact_grid, levels=100, cmap='jet')
plt.colorbar()
plt.title("Exact Solution")

# Predicted solution
plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_pred_grid, levels=100, cmap='jet')
plt.colorbar()
plt.title("Predicted (PINN)")

# Absolute Error
plt.subplot(1, 3, 3)
plt.contourf(X, Y, abs_error_grid, levels=100, cmap='hot')
plt.colorbar()
plt.title("Absolute Error")

plt.tight_layout()
plt.savefig("core/output/poissons2d/result_u_pred.png")
