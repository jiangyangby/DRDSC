from __future__ import print_function, absolute_import, division

import tensorflow as tf
from tensorflow.contrib import layers

mu = 1.0e-6


@tf.custom_gradient
def f_norm(x):
    f2 = tf.square(tf.norm(x, ord='fro', axis=[-2, -1]))
    f = tf.sqrt(f2 + mu ** 2) - mu
    def grad(dy):
        return dy * (x / tf.sqrt(f2 + mu ** 2))
    return f, grad


@tf.custom_gradient
def l2_norm(x):
    f2 = tf.square(tf.norm(x, ord=2))
    f = tf.sqrt(f2 + mu ** 2) - mu
    def grad(dy):
        return dy * (x / tf.sqrt(f2 + mu ** 2))
    return f, grad


class RSCConvAE:
    '''
    Duet Robust Deep Subspace Clustering
    '''

    def __init__(self, n_input, kernel_size, n_hidden, z_dim, lamda1=1.0,
                 lamda2=1.0, eta1=1.0, eta2=1.0, batch_size=200, reg=None,
                 denoise=False, save_path=None, restore_path=None,
                 normalize_input=False, logs_path='./logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.reg = reg
        self.save_path = save_path
        self.restore_path = restore_path
        self.iter = 0

        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])

        weights = self._initialize_weights()
        self.x_noise = weights['x_noise']
        self.z_noise = weights['z_noise']

        self.z, self.Coef, self.x_r, self.x_diff, self.z_diff = \
            self._forward(denoise, normalize_input, weights)

        # l_2 reconstruction loss
        self.reconst_cost = self._get_reconstruction_loss(eta1)
        tf.summary.scalar("recons_loss", self.reconst_cost)

        self.reg_loss = self._get_coef_reg_loss(reg_type='l2')  # l2 reg
        tf.summary.scalar("reg_loss", lamda2 * self.reg_loss)

        selfexpress_cost = tf.square(self.z_diff - self.z_noise)
        z_noise_reg = tf.map_fn(lambda frame: l2_norm(frame), self.z_noise)
        self.selfexpress_loss = 0.5 * \
            tf.reduce_sum(selfexpress_cost) + eta2 * tf.reduce_sum(z_noise_reg)
        tf.summary.scalar("selfexpress_loss", lamda1 *
                          self.selfexpress_loss)

        self.loss = self.reconst_cost + lamda1 * \
            self.selfexpress_loss + lamda2 * self.reg_loss

        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(
            # self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver(
            [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        self.summary_writer = tf.summary.FileWriter(
            logs_path, graph=tf.get_default_graph())

    def _build_input(self, denoise, normalize_input):
        if not normalize_input:
            x_input = self.x
        else:
            x_input = tf.map_fn(
                lambda frame: tf.image.per_image_standardization(frame), self.x)
        if denoise:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
        return x_input

    def _forward(self, denoise, normalize_input, weights):
        x_input = self._build_input(denoise, normalize_input)
        latent, shape = self.encoder(x_input, weights)

        z = tf.reshape(latent, [self.batch_size, -1])
        Coef = weights['Coef']
        Coef = Coef - tf.diag(tf.diag_part(Coef))
        z_c = tf.matmul(Coef, z)
        latent_c = tf.reshape(z_c, tf.shape(latent))
        x_r = self.decoder(latent_c, weights, shape)
        z_diff = z - z_c
        x_diff = x_input - x_r
        return z, Coef, x_r, x_diff, z_diff

    def _get_reconstruction_loss(self, eta1):
        reconst_cost = tf.square(self.x_diff - self.x_noise)  # l2
        x_noise_3dim = tf.squeeze(self.x_noise)
        x_noise_group_reg = tf.map_fn(
            lambda frame: f_norm(frame), x_noise_3dim)
        reconst_cost = 0.5 * tf.reduce_sum(reconst_cost) + \
            eta1 * tf.reduce_sum(x_noise_group_reg)
        return reconst_cost

    def _get_coef_reg_loss(self, reg_type='l2'):
        if reg_type is 'l2':
            loss = tf.reduce_sum(tf.square(self.Coef))
        elif reg_type is 'l1':
            loss = tf.reduce_sum(tf.abs(self.Coef))
        return loss

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        # all_weights['Coef'] = tf.Variable(
        #     tf.random_normal([self.batch_size, self.batch_size],
        #                      mean=0.0, stddev=0.1, dtype=tf.float32,
        #                      seed=None), name='Coef')
        all_weights['Coef'] = tf.Variable(
            0 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
        all_weights['x_noise'] = tf.Variable(
            tf.zeros([self.batch_size, self.n_input[0],
                      self.n_input[1], 1], tf.float32), name='Coef')
        all_weights['z_noise'] = tf.Variable(
            tf.zeros([self.batch_size, self.z_dim], tf.float32), name='Coef')

        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(
            tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        for iter_i in range(1, n_layers):
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i], self.n_hidden[iter_i - 1],
                                                                           self.n_hidden[iter_i]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(
                tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))

        for iter_i in range(1, n_layers):
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i], self.kernel_size[n_layers - iter_i],
                                                                           self.n_hidden[n_layers - iter_i - 1], self.n_hidden[n_layers - iter_i]],
                                                       initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(tf.zeros(
                [self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))

        dec_name_wi = 'dec_w' + str(n_layers - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                   initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        dec_name_bi = 'dec_b' + str(n_layers - 1)
        all_weights[dec_name_bi] = tf.Variable(
            tf.zeros([1], dtype=tf.float32))

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[
                                1, 2, 2, 1], padding='SAME'), weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())

        for iter_i in range(1, len(self.n_hidden)):
            layeri = tf.nn.bias_add(tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[
                                    1, 2, 2, 1], padding='SAME'), weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())

        layer3 = layeri
        return layer3, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        for iter_i in range(n_layers):
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]),
                                                   strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
        return layer3

    def partial_fit(self, X, lr):
        cost, summary, _, Coef, z_diff, x_diff = self.sess.run(
            (self.loss, self.merged_summary_op, self.optimizer, self.Coef,
             self.z_diff, self.x_diff),
            feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef, z_diff, x_diff

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.save_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")
