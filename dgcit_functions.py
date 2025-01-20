import tensorflow as tf
import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import KFold
import os
logging.getLogger('tensorflow').disabled = True
tf.keras.backend.set_floatx('float64')

# Set the environment variable to make only the first 4 GPUs visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def dgcit(x, y, z, seed=0, batch_size=64, n_iter=1000, train_writer=None, current_iters=0, M=500, k=2, b=30, j=1000):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    x = x.cpu().numpy().astype(np.float64)
    y = y.cpu().numpy().astype(np.float64)
    z = z.cpu().numpy().astype(np.float64)
    n, z_dim = z.shape
    _, y_dims = y.shape
    _, x_dims = x.shape
    # no. of random and hidden dimensions
    if z_dim <= 20:
        v_dims = int(3)
        h_dims = int(3)
    else:
        v_dims = int(50)
        h_dims = int(512)

    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0 / 3.0))
    # create instance of G & D
    lr = 0.0005
    generator_x = WGanGenerator(n, z_dim, h_dims, v_dims, x_dims, batch_size)
    generator_y = WGanGenerator(n, z_dim, h_dims, v_dims, y_dims, batch_size)
    discriminator_x = WGanDiscriminator(n, z_dim, h_dims, x_dims, batch_size)
    discriminator_y = WGanDiscriminator(n, z_dim, h_dims, y_dims, batch_size)

    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0
    w_clipping_val = 0.5
    w_clipping_norm = 1.0
    scaling_coef = 1.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30

    gx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
    gy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    @tf.function
    def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_x.call(gen_inputs)
        fake_x_p = generator_x.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_x.trainable_variables)
        dx_optimiser.apply_gradients(zip(d_grads, discriminator_x.trainable_variables))

    @tf.function
    def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_x.call(gen_inputs)
            fake_x_p = generator_x.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_x.trainable_variables)
        gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))
        return gen_loss

    @tf.function
    def y_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_y.call(gen_inputs)
        fake_x_p = generator_y.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_y.call(d_real)
            f_fake = discriminator_y.call(d_fake)
            f_real_p = discriminator_y.call(d_real_p)
            f_fake_p = discriminator_y.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_y.trainable_variables)
        dy_optimiser.apply_gradients(zip(d_grads, discriminator_y.trainable_variables))

    @tf.function
    def y_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_y.call(gen_inputs)
            fake_x_p = generator_y.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_y.call(d_real)
            f_fake = discriminator_y.call(d_fake)
            f_real_p = discriminator_y.call(d_real_p)
            f_fake_p = discriminator_y.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_y.trainable_variables)
        gy_optimiser.apply_gradients(zip(generator_grads, generator_y.trainable_variables))
        return gen_loss

    psy_x_all = []
    phi_x_all = []
    psy_y_all = []
    phi_y_all = []
    test_samples = b
    test_size = int(n/k)

    # split the train-test sets to k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    epochs = int(n_iter)

    for train_idx, test_idx in kf.split(x):
        x_train, y_train, z_train = x[train_idx], y[train_idx], z[train_idx]

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train,
                                                      z_train))
        # Repeat n epochs
        training = dataset.repeat(epochs)
        training_dataset = training.shuffle(100).batch(batch_size * 2)
        # test-set is the one left
        testing_dataset = tf.data.Dataset.from_tensor_slices((x[test_idx], y[test_idx], z[test_idx]))

        for x_batch, y_batch, z_batch in training_dataset.take(n_iter):
            if x_batch.shape[0] != batch_size * 2:
                continue

            # seperate the batch into two parts to train two gans
            x_batch1 = x_batch[:batch_size, ...]
            x_batch2 = x_batch[batch_size:, ...]
            y_batch1 = y_batch[:batch_size, ...]
            y_batch2 = y_batch[batch_size:, ...]
            z_batch1 = z_batch[:batch_size, ...]
            z_batch2 = z_batch[batch_size:, ...]

            noise_v = v_dist.sample([batch_size, v_dims])
            noise_v = tf.cast(noise_v, tf.float64)
            noise_v_p = v_dist.sample([batch_size, v_dims])
            noise_v_p = tf.cast(noise_v_p, tf.float64)
            x_update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
            loss_x = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
            y_update_d(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
            loss_y = y_update_g(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)

            with train_writer.as_default():
                # tf.summary.scalar('Wasserstein X Discriminator Loss', x_disc_loss, step=current_iters)
                tf.summary.scalar('Wasserstein X GEN Loss', loss_x, step=current_iters)
                # tf.summary.scalar('Wasserstein Y Discriminator Loss', y_disc_loss, step=current_iters)
                tf.summary.scalar('Wasserstein Y GEN Loss', loss_y, step=current_iters)
                train_writer.flush()

            current_iters += 1

        psy_x_b = []
        phi_x_b = []
        psy_y_b = []
        phi_y_b = []

        x_samples = []
        y_samples = []
        z = []
        x = []
        y = []

        # the following code generate x_1, ..., x_400 for all B and it takes 61 secs for one test
        for test_x, test_y, test_z in testing_dataset:
            tiled_z = tf.tile(test_z, [M, 1])
            noise_v = v_dist.sample([M, v_dims])
            noise_v = tf.cast(noise_v, tf.float64)
            g_inputs = tf.concat([tiled_z, noise_v], axis=1)
            # generator samples from G and evaluate from D
            fake_x = generator_x.call(g_inputs, training=False)
            fake_y = generator_y.call(g_inputs, training=False)
            x_samples.append(fake_x)
            y_samples.append(fake_y)
            z.append(test_z)
            x.append(test_x)
            y.append(test_y)

        standardise = True

        if standardise:
            x_samples = (x_samples - tf.reduce_mean(x_samples)) / tf.math.reduce_std(x_samples)
            y_samples = (y_samples - tf.reduce_mean(y_samples)) / tf.math.reduce_std(y_samples)
            x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
            y = (y - tf.reduce_mean(y)) / tf.math.reduce_std(y)
            z = (z - tf.reduce_mean(z)) / tf.math.reduce_std(z)

        f1 = CharacteristicFunction(M, x_dims, z_dim, test_size)
        f2 = CharacteristicFunction(M, y_dims, z_dim, test_size)
        for i in range(test_samples):
            phi_x = tf.reduce_mean(f1.call(x_samples, z), axis=1)
            phi_y = tf.reduce_mean(f2.call(y_samples, z), axis=1)
            psy_x = tf.squeeze(f1.call(x, z))
            psy_y = tf.squeeze(f2.call(y, z))

            psy_x_b.append(psy_x)
            phi_x_b.append(phi_x)
            psy_y_b.append(psy_y)
            phi_y_b.append(phi_y)
            f1.update()
            f2.update()

        psy_x_all.append(psy_x_b)
        phi_x_all.append(phi_x_b)
        psy_y_all.append(psy_y_b)
        phi_y_all.append(phi_y_b)

    # reshape
    psy_x_all = tf.reshape(psy_x_all, [k, test_samples, test_size])
    psy_y_all = tf.reshape(psy_y_all, [k, test_samples, test_size])
    phi_x_all = tf.reshape(phi_x_all, [k, test_samples, test_size])
    phi_y_all = tf.reshape(phi_y_all, [k, test_samples, test_size])
    
    t_b = 0.0
    std_b = 0.0
    for n in range(k):
        t, std = t_and_sigma(psy_x_all[n], psy_y_all[n], phi_x_all[n], phi_y_all[n])
        t_b += t
        std_b += std
    t_b = t_b / tf.cast(k, tf.float64)
    std_b = std_b / tf.cast(k, tf.float64)

    psy_x_all = tf.transpose(psy_x_all, (1, 0, 2))
    psy_y_all = tf.transpose(psy_y_all, (1, 0, 2))
    phi_x_all = tf.transpose(phi_x_all, (1, 0, 2))
    phi_y_all = tf.transpose(phi_y_all, (1, 0, 2))

    psy_x_all = tf.reshape(psy_x_all, [test_samples, test_size*k])
    psy_y_all = tf.reshape(psy_y_all, [test_samples, test_size*k])
    phi_x_all = tf.reshape(phi_x_all, [test_samples, test_size*k])
    phi_y_all = tf.reshape(phi_y_all, [test_samples, test_size*k])

    stat, critical_vals = test_statistics(psy_x_all, psy_y_all, phi_x_all, phi_y_all, t_b, std_b, j)
    comparison = [c > stat or c == stat for c in critical_vals]
    comparison = np.reshape(comparison, (-1,))
    p_value = np.sum(comparison.astype(np.float64)) / j
    return p_value


class WGanGenerator(tf.keras.Model):
    '''
    class for WGAN generator
    Args:
        inputs, noise and confounding factor [v, z], of shape [batch size, z_dims + v_dims]
    return:
       fake samples of shape [batch size, x_dims]
    '''
    def __init__(self, n_samples, z_dims, h_dims, v_dims, x_dims, batch_size):
        super(WGanGenerator, self).__init__()
        self.n_samples = n_samples
        self.hidden_dims = h_dims
        self.batch_size = batch_size
        self.dz = z_dims
        self.dx = x_dims
        self.dv = v_dims

        self.input_dim = self.dz + self.dv
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, self.dx]

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        out = tf.math.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        return out


class WGanDiscriminator(tf.keras.Model):
    '''
    class for WGAN discriminator
    Args:
        inputss: real and fake samples of shape [batch size, x_dims]
    return:
       features f_x of shape [batch size, features]
    '''
    def __init__(self, n_samples, z_dims, h_dims, v_dims, batch_size):
        super(WGanDiscriminator, self).__init__()
        self.n_samples = n_samples
        self.hidden_dims = h_dims
        self.batch_size = batch_size

        self.input_dim = z_dims + v_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, -1])
        z = tf.cast(z, tf.float64)
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.sigmoid(tf.matmul(h1, self.w2) + self.b2)
        # out = tf.nn.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        out = tf.matmul(h1, self.w3) + self.b3
        return out


class MINEDiscriminator(tf.keras.layers.Layer):
    '''
    class for MINE discriminator for benchmark GCIT
    '''

    def __init__(self, in_dims, output_activation='linear'):
        super(MINEDiscriminator, self).__init__()
        self.output_activation = output_activation
        self.input_dim = in_dims

        self.w1a = self.xavier_var_creator()
        self.w1b = self.xavier_var_creator()
        self.b1 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

        self.w2a = self.xavier_var_creator()
        self.w2b = self.xavier_var_creator()
        self.b2 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

        self.w3 = self.xavier_var_creator()
        self.b3 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

    def xavier_var_creator(self):
        xavier_stddev = 1.0 / tf.sqrt(self.input_dim / 2.0)
        init = tf.random.normal(shape=[self.input_dim, ], mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(self.input_dim, ), trainable=True)
        return var

    def mine_layer(self, x, x_hat, wa, wb, b):
        return tf.math.tanh(wa * x + wb * x_hat + b)

    def call(self, x, x_hat):
        h1 = self.mine_layer(x, x_hat, self.w1a, self.w1b, self.b1)
        h2 = self.mine_layer(x, x_hat, self.w2a, self.w2b, self.b2)
        out = self.w3 * (h1 + h2) + self.b3
        return out, tf.exp(out)


class CharacteristicFunction:
    '''
    class to construct a function that represents the characteristic function
    '''

    def __init__(self, size, x_dims, z_dims, test_size):
        self.n_samples = size
        self.hidden_dims = 20
        self.test_size = test_size

        self.input_dim = z_dims + x_dims
        self.z_dims = z_dims
        self.x_dims = x_dims
        self.input_shape1x = [self.x_dims, self.hidden_dims]
        self.input_shape1z = [self.z_dims, self.hidden_dims]
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, 1]

        self.w1x = self.xavier_var_creator(self.input_shape1x)
        self.b1 = tf.squeeze(self.xavier_var_creator([self.hidden_dims, 1]))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = tf.sqrt(2.0 / (input_shape[0]))
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def update(self):
        self.w1x = self.xavier_var_creator(self.input_shape1x)
        self.b1 = tf.squeeze(self.xavier_var_creator([self.hidden_dims, 1]))
        self.w2 = self.xavier_var_creator(self.input_shape2)

    def call(self, x, z):
        # inputs are concatenations of z and v
        x = tf.reshape(tensor=x, shape=[self.test_size, -1, self.x_dims])
        z = tf.reshape(tensor=z, shape=[self.test_size, -1, self.z_dims])
        # we asssume parameter b for z to be 0
        h1 = tf.nn.sigmoid(tf.matmul(x, self.w1x) + self.b1)
        out = tf.nn.sigmoid(tf.matmul(h1, self.w2))
        return out


def t_and_sigma(psy_x_i, psy_y_i, phi_x_i, phi_y_i):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    t_b = tf.reduce_sum(matrix, axis=1) / tf.cast(n, tf.float64)
    t_b = tf.expand_dims(t_b, axis=1)

    crit_matrix = matrix - t_b
    std_b = tf.sqrt(tf.reduce_sum(crit_matrix**2, axis=1) / tf.cast(n-1, tf.float64))
    return t_b, std_b


def test_statistics(psy_x_i, psy_y_i, phi_x_i, phi_y_i, t_b, std_b, j):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    crit_matrix = matrix - t_b
    test_stat = tf.reduce_max(tf.abs(tf.sqrt(tf.cast(n, tf.float64)) * tf.squeeze(t_b) / std_b))

    sig = tf.reduce_sum(crit_matrix[None, :, :] * crit_matrix[:, None, :], axis=2)
    coef = std_b[None, :] * std_b[:, None] * tf.cast(n-1, tf.float64)
    sig_xy = sig / coef

    eigenvalues, eigenvectors = tf.linalg.eigh(sig_xy)
    base = tf.zeros_like(eigenvectors)
    eig_vals = tf.sqrt(eigenvalues + 1e-12)
    lamda = tf.linalg.set_diag(base, eig_vals)
    sig_sqrt = tf.matmul(tf.matmul(eigenvectors, lamda), tf.linalg.inv(eigenvectors))

    z_dist = tfp.distributions.Normal(0.0, scale=1.0)
    z_samples = z_dist.sample([b*b, j])
    z_samples = tf.cast(z_samples, tf.float64)
    vals = tf.matmul(sig_sqrt, z_samples)
    t_j = tf.reduce_max(vals, axis=0)
    return test_stat, t_j


def benchmark_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l, xp=None, yp=None):
    '''
    :param x: real data of shape [batch size, sequence length]
    :param y: fake data of shape [batch size, sequence length]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and several values for monitoring the training process)
    '''
    if yp is None:
        yp = y
    if xp is None:
        xp = x
    x = tf.reshape(x, [x.shape[0], -1])
    y = tf.reshape(y, [y.shape[0], -1])
    xp = tf.reshape(xp, [xp.shape[0], -1])
    yp = tf.reshape(yp, [yp.shape[0], -1])
    loss_xy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = benchmark_sinkhorn(x, xp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = benchmark_sinkhorn(y, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy - 0.5 * loss_xx - 0.5 * loss_yy

    return loss


def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10):
    '''
    :param x: a tensor of shape [batch_size, sequence length]
    :param y: a tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :param epsilon: (float) entropic regularity constant
    :param L: (int) number of iterations
    :return: V: (float) value of regularized optimal transport
    '''
    n_data = x.shape[0]
    # Note that batch size of x can be different from batch size of y
    m = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    n = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    m = tf.expand_dims(m, axis=1)
    n = tf.expand_dims(n, axis=1)

    c_xy = cost_xy(x, y, scaling_coef)  # shape: [batch_size, batch_size]

    k = tf.exp(-c_xy / epsilon) + 1e-09  # add 1e-09 to prevent numerical issues
    k_t = tf.transpose(k)

    a = tf.expand_dims(tf.ones(n_data, dtype=tf.float64), axis=1)
    b = tf.expand_dims(tf.ones(n_data, dtype=tf.float64), axis=1)

    for i in range(L):
        b = m / tf.matmul(k_t, a)  # shape: [m,]
        a = n / tf.matmul(k, b)  # shape: [m,]

    return tf.reduce_sum(a * k * tf.reshape(b, (1, -1)) * c_xy)


def cost_xy(x, y, scaling_coef):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, x dims]
    :param y: y is tensor of shape [batch_size, y dims]
    :param scaling_coef: a scaling coefficient for distance between x and y
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    return tf.reduce_sum((x - y)**2, -1) * scaling_coef