import tensorflow as tf
import numpy as np


class pde_loss_poissions2d:
    # def __init__(self,pred_nn, input):
    #     self.pred_nn = pred_nn 
    #     self.input = input

    def forcing_term(input):
        x, y = tf.expand_dims(input[:, 0], 1), tf.expand_dims(input[:, 1], 1)
        return -tf.sin(np.pi * x) * tf.sin(np.pi * y)

    def physics_loss(self, model, input):
        x, y = tf.expand_dims(input[:, 0], 1), tf.expand_dims(input[:, 1], 1)
        with tf.GradientTape(presistent=True) as tape1:
            tape1.watch([x,y])                                                              
            with tf.GradientTape(presistent=True) as tape2:
                tape2.watch([x,y])
                u = model(tf.concat([x,y], axis=1))
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u_y)
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1. grdient(u_y, y)
        del tape1, tape2

        residual = -1.0 * (u_xx + u_yy) - pde_loss_poissions2d.forcing_term(input)

        return tf.reduce_mean(tf.square(residual))
    
    def boundary_loss(model, input_boundary):
        u_pred = model(input_boundary)
        return tf.reduce_mean(tf.square(u_pred))

# def pde_loss_poissions2d(
#         pred_nn: tf.Tensor,
#         pred_grad_x_nn: tf.Tensor,
#         pred_grad_y_nn: tf.Tensor,
#         pred_grad_xx_nn: tf.Tensor,
#         pred_grad_yy_nn: tf.Tensor
# ):
#     pde_diffusion = -1.0 * (pred_grad_xx_nn + pred_grad_yy_nn)

#     # tf.print(f"Shape of pde_diffusion {pde_diffusion.shape}")

#     residual = pde_diffusion

#     pde_residual = tf.reduce_mean(tf.square(residual))
#     return pde_residual