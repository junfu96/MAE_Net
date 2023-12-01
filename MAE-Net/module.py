import tensorflow as tf
import tensorflow_addons as tfa

from layers import Oper2D, Oper2DTranspose


# mae_net models.

def OpGenerator(input_shape=(256, 256, 3), n=1):
    dim = 64
    Norm = tfa.layers.InstanceNormalization

    def _residual_block(x):
        dim = x.shape[-1]
        x1 = tf.nn.tanh(x)
        h = x1

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        # SNP
        h = Norm()(h)
        h = tf.nn.tanh(h)
        h = Oper2D(dim, 3, n=n, padding='valid', use_bias=False)(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        h = Norm()(h)
        h = tf.nn.tanh(h)
        h = Oper2D(dim, 3, n=n, padding='valid', use_bias=False)(h)

        return tf.nn.tanh(tf.keras.layers.add([x, h]))

    h = inputs = tf.keras.Input(shape=input_shape)

    dim *= 2
    h = Oper2D(dim, 3, strides=2, n=n, padding='same', use_bias=False)(h)
    h = Norm()(h)

    h = _residual_block(h)

    # Classification branch.
    x = tf.keras.layers.MaxPool2D(h.shape[1])(h)
    x = tf.keras.layers.Flatten()(x)
    y_class = tf.keras.layers.Dense(2, activation='softmax')(x)

    dim //= 2
    h = Oper2DTranspose(3, 3, strides=2, n=n, padding='same')(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=[h, y_class])


def OpDiscriminator(input_shape=(256, 256, 3), n=1):
    dim = 64
    Norm = tfa.layers.InstanceNormalization

    h = inputs = tf.keras.Input(shape=input_shape)

    # SNP
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    h = Oper2D(dim, 4, strides=2, n=n, padding='same')(h)

    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    h = Oper2D(2 * dim, 4, strides=4, n=n, padding='same', use_bias=False)(h)

    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    h = Oper2D(1, 4, strides=1, n=n, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)


def ConvGenerator(input_shape=(256, 256, 3)):
    Norm = tfa.layers.InstanceNormalization
    dim = 64
    n_blocks = 9
    n_downsamplings = 2

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        # SNP
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)

        return tf.keras.layers.add([x, h])

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')

    # 1-SNP
    h = Norm()(h)
    h = tf.nn.relu(h)
    h = tf.keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2

        # 2-SNP
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

        if _ == 4:
            x = tf.keras.layers.MaxPool2D(h.shape[1])(h)
            x = tf.keras.layers.Flatten()(x)
            y_class = tf.keras.layers.Dense(2, activation='softmax')(x)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        # 4-SNP
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')

    # 5-SNP
    h = Norm()(h)
    h = tf.tanh(h)
    h = tf.keras.layers.Conv2D(3, 7, padding='valid')(h)

    return tf.keras.Model(inputs=inputs, outputs=[h, y_class])


def ConvDiscriminator(input_shape=(256, 256, 3), n_downsamplings=3):
    dim = 64
    dim_ = dim
    Norm = tfa.layers.InstanceNormalization

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)


def ConvCompGenerator(input_shape=(256, 256, 3)):
    dim = 64
    Norm = tfa.layers.InstanceNormalization

    def _residual_block(x):
        dim = x.shape[-1]
        h = x
        # SNP
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)

        return tf.keras.layers.add([x, h])

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 2
    dim *= 2

    # 2-SNP
    h = Norm()(h)
    h = tf.nn.relu(h)
    h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)

    # 3
    h = _residual_block(h)

    x = tf.keras.layers.MaxPool2D(h.shape[1])(h)
    x = tf.keras.layers.Flatten()(x)
    y_class = tf.keras.layers.Dense(2, activation='softmax')(x)

    # 4
    dim //= 2

    # 4-SNP
    h = Norm()(h)
    h = tf.nn.tanh(h)
    h = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=[h, y_class])


def ConvCompDiscriminator(input_shape=(256, 256, 3)):
    dim = 64
    dim_ = dim
    Norm = tfa.layers.InstanceNormalization

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1

    # 1-SNP
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)

    # SNP
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    h = tf.keras.layers.Conv2D(2 * dim, 4, strides=4, padding='same', use_bias=False)(h)

    # Classification and Real or Fake
    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)


#  Linear dropping learning rate scheduler.

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                    1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
