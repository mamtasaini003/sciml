import tensorflow as tf

class PINN(tf.keras.Model):

    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(units, activation='tanh') for units in layers[1:-1]]
        self.output_layer = tf.keras.layers.Dense(layers[-1])
        self.build(input_shape=(None, layers[0]))

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


