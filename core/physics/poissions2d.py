import tensorflow as tf
import numpy as np


class pde_loss_poissions2d:
    # def __init__(self,pred_nn, input):
    #     self.pred_nn = pred_nn 
    #     self.input = input
    @staticmethod
    def forcing_term(input):
        x, y = tf.expand_dims(input[:, 0], 1), tf.expand_dims(input[:, 1], 1)
        return -tf.math.sin(np.pi * x) * tf.math.sin(np.pi * y)
    
    @staticmethod
    def physics_loss(model, input):
        x, y = tf.expand_dims(input[:, 0], 1), tf.expand_dims(input[:, 1], 1)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x,y])                                                              
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([x,y])
                u = model(tf.concat([x,y], axis=1))
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1. gradient(u_y, y)
        del tape1, tape2

        residual = (u_xx + u_yy) - pde_loss_poissions2d.forcing_term(input)

        return tf.reduce_mean(tf.square(residual))
    
    @staticmethod    
    def boundary_loss(model, input_boundary):
        u_pred = model(input_boundary)
        return tf.reduce_mean(tf.square(u_pred))
