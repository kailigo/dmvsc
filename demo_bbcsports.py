import tensorflow as tf
import numpy as np
import argparse
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from numpy.linalg import svd
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres

import matplotlib.pyplot as plt


def next_batch(data, _index_in_epoch, batch_size, _epochs_completed):
    _num_examples = data.shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        # label = label[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed



class ConvAE1(object):
    def __init__(self, feature_size, n_hidden, learning_rate=1e-3, batch_size=256, \
                 reg=None, denoise=False, model_path=None, restore_path=None,
                 logs_path='./models_face'):

        # n_hidden is a arrary contains the number of neurals on every layer
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.batch_size = batch_size
        self.iter = 0
        self.feature_size = feature_size
        weights = self._initialize_weights()

        # model
        # self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.x = tf.placeholder(tf.float32, [None, feature_size])

        if denoise == False:
            x_input = self.x
            latent = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        # self.z = tf.reshape(latent, [batch_size, -1])
        # self.latent = latent
        self.x_r = self.decoder(latent, weights)
        self.saver = tf.train.Saver()
        # cost for reconstruction
        # l_2 loss
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))  # choose crossentropy or l2 loss
        tf.summary.scalar("l2_loss", self.cost)
        self.merged_summary_op = tf.summary.merge_all()
        self.loss = self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        # feat_size1 = self.n_input[0] * self.n_input[1]
        all_weights['v1_enc_w0'] = tf.Variable(tf.random_normal([self.feature_size, self.n_hidden[0]]),
                                               name='v1_enc_w0')
        all_weights['v1_enc_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]), name='v1_enc_b0')
        all_weights['v1_enc_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.n_hidden[1]]), name='v1_enc_w1')
        all_weights['v1_enc_b1'] = tf.Variable(tf.random_normal([self.n_hidden[1]]), name='v1_enc_b1')

        all_weights['v1_dec_w0'] = tf.Variable(tf.random_normal([self.n_hidden[1], self.n_hidden[0]]), name='v1_dec_w0')
        all_weights['v1_dec_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]), name='v1_dec_b0')
        all_weights['v1_dec_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.feature_size]),
                                               name='v1_dec_w1')
        all_weights['v1_dec_b1'] = tf.Variable(tf.random_normal([self.feature_size]), name='v1_dec_b1')

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        v1_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['v1_enc_w0']), weights['v1_enc_b0']))
        v1_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(v1_layer1, weights['v1_enc_w1']), weights['v1_enc_b1']))

        return v1_layer2

    # Building the decoder
    def decoder(self, z, weights):

        v1_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(z, weights['v1_dec_w0']), weights['v1_dec_b0']))
        v1_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(v1_layer_1, weights['v1_dec_w1']), weights['v1_dec_b1']))

        return v1_layer_2

    def partial_fit(self, X):
        cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict={self.x: X})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")


class ConvAE2(object):
    def __init__(self, feature_size, n_hidden, learning_rate=1e-3, batch_size=256, \
                 reg=None, denoise=False, model_path=None, restore_path=None,
                 logs_path='./models_face'):
        # n_hidden is a arrary contains the number of neurals on every layer
        # self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        # self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.iter = 0
        self.feature_size = feature_size
        weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(tf.float32, [None, feature_size])

        if denoise == False:
            x_input = self.x
            latent = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent = self.encoder(x_input, weights)
        # self.z = tf.reshape(latent, [batch_size, -1])
        # self.latent = latent
        self.x_r = self.decoder(latent, weights)
        self.saver = tf.train.Saver()
        # cost for reconstruction
        # l_2 loss
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))  # choose crossentropy or l2 loss
        tf.summary.scalar("l2_loss", self.cost)

        self.merged_summary_op = tf.summary.merge_all()
        self.loss = self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            self.loss)  # GradientDescentOptimizer #AdamOptimizer
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['v2_enc_w0'] = tf.Variable(tf.random_normal([self.feature_size, self.n_hidden[0]]), name='v2_enc_w0')
        all_weights['v2_enc_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]), name='v2_enc_b0')
        all_weights['v2_enc_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.n_hidden[1]]), name='v2_enc_w1')
        all_weights['v2_enc_b1'] = tf.Variable(tf.random_normal([self.n_hidden[1]]), name='v2_enc_b1')

        all_weights['v2_dec_w0'] = tf.Variable(tf.random_normal([self.n_hidden[1], self.n_hidden[0]]), name='v2_dec_w0')
        all_weights['v2_dec_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]), name='v2_dec_b0')
        all_weights['v2_dec_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.feature_size]), name='v2_dec_w1')
        all_weights['v2_dec_b1'] = tf.Variable(tf.random_normal([self.feature_size]), name='v2_dec_b1')

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        v2_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['v2_enc_w0']), weights['v2_enc_b0']))
        v2_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(v2_layer1, weights['v2_enc_w1']), weights['v2_enc_b1']))
        return v2_layer2

    # Building the decoder
    def decoder(self, z, weights):
        # Encoder Hidden layer with relu activation #1
        v2_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(z, weights['v2_dec_w0']), weights['v2_dec_b0']))
        v2_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(v2_layer_1, weights['v2_dec_w1']), weights['v2_dec_b1']))

        return v2_layer_2

    def partial_fit(self, X):
        cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict={self.x: X})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")


def ae_feature_clustering(CAE, X):
    CAE.restore()

    Z = CAE.transform(X)
    sio.savemat('AE_YaleB.mat', dict(Z=Z))

    return



def test_face_pretrain(Img, CAE, n_input):
    batch_x_test = Img[200:300, :]
    batch_x_test = np.reshape(batch_x_test, [100, n_input[0], n_input[1], 1])
    CAE.restore()
    x_re = CAE.reconstruct(batch_x_test)

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(batch_x_test[i, :, :, 0], vmin=0, vmax=255, cmap="gray")  #
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_re[i, :, :, 0], vmin=0, vmax=255, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
        plt.tight_layout()
    plt.show()
    return


class dmvsc(object):
    def __init__(self, feature_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0, batch_size=200, \
                 model_path=None, restore_path=None, logs_path='./logs', pretrain_model_path1=None, pretrain_model_path2=None, init_coef_file=None):

        # self.n_input = n_input
        # self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        # self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        self.pretrain_model_path1 = pretrain_model_path1
        self.pretrain_model_path2 = pretrain_model_path2
        self.feature_size = feature_size
        self.init_coef_file = init_coef_file


        denoise = False
        
        # input required to be fed
        # self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])

        self.x1 = tf.placeholder(tf.float32, [None, feature_size[0]])
        self.x2 = tf.placeholder(tf.float32, [None, feature_size[1]])
        self.learning_rate = tf.placeholder(tf.float32, [])

        weights = self._initialize_weights1()

        if denoise == False:
            x_input1 = self.x1
            x_input2 = self.x2
            # view = tf.gather(self.x, i)  # NxWxHxC
            # v1_latent, v2_latent = self.encoder(x_input1, x_input2, weights)

            v1_latent_pre, v1_latent, v2_latent_pre, v2_latent = \
                self.encoder1(x_input1, x_input2, weights)

            # latent2, shape2 = self.encoder(x_input, weights)
        else:
            x_input1 = tf.add(self.x1, tf.random_normal(shape=tf.shape(self.x1),
                                                        mean=0,
                                                        stddev=0.2,
                                                        dtype=tf.float32))
            x_input2 = tf.add(self.x2, tf.random_normal(shape=tf.shape(self.x2),
                                                        mean=0,
                                                        stddev=0.2,
                                                        dtype=tf.float32))
            v1_latent, v2_latent = self.encoder(x_input1, x_input2, weights)


        v1_z_pre = tf.reshape(v1_latent_pre, [batch_size, -1])
        coef1 = weights['coef1']
        v1_z_c_pre = tf.matmul(coef1, v1_z_pre)
        v1_latent_c_pre = tf.reshape(v1_z_c_pre, tf.shape(v1_latent_pre))  # petential problem here
        self.v1_z_pre = v1_z_pre

        v2_z_pre = tf.reshape(v2_latent_pre, [batch_size, -1])
        coef2 = weights['coef2']
        v2_z_c_pre = tf.matmul(coef2, v2_z_pre)
        v2_latent_c_pre = tf.reshape(v2_z_c_pre, tf.shape(v2_latent_pre))  # petential problem here
        self.v2_z_pre = v2_z_pre


        v1_z = tf.reshape(v1_latent, [batch_size, -1])
        coef1 = weights['coef1']
        v1_z_c = tf.matmul(coef1, v1_z)
        self.coef1 = coef1
        v1_latent_c = tf.reshape(v1_z_c, tf.shape(v1_latent))  # petential problem here
        self.v1_z = v1_z

        v2_z = tf.reshape(v2_latent, [batch_size, -1])
        coef2 = weights['coef2']
        v2_z_c = tf.matmul(coef2, v2_z)
        self.coef2 = coef2
        v2_latent_c = tf.reshape(v2_z_c, tf.shape(v2_latent))  # petential problem here
        self.v2_z = v2_z

        self.v1_x_r, self.v2_x_r = self.decoder(v1_latent_c, v2_latent_c, weights)
        # l_2 reconstruction loss
        v1_reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.v1_x_r, self.x1), 2.0))
        v2_reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.v2_x_r, self.x2), 2.0))
        self.reconst_cost = v1_reconst_cost + v2_reconst_cost
        tf.summary.scalar("recons_loss", self.reconst_cost)

        self.con_Coef = weights['con_Coef']
        coef_diff_loss1 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(coef1, self.con_Coef), 2.0))
        coef_diff_loss2 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(coef2, self.con_Coef), 2.0))
        self.coef_diff_loss = tf.add(coef_diff_loss1, coef_diff_loss2)
        tf.summary.scalar("coef_diff_loss", re_constant3 * self.coef_diff_loss)

        self.reg_losses = tf.reduce_sum(tf.pow(self.con_Coef, 2.0))
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        v1_selfexpress_losses_pre = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(v1_z_c_pre, v1_z_pre), 2.0))
        v2_selfexpress_losses_pre = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(v2_z_c_pre, v2_z_pre), 2.0))

        v1_selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(v1_z_c, v1_z), 2.0))
        v2_selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(v2_z_c, v2_z), 2.0))
        # self.selfexpress_losses = tf.add(v1_selfexpress_losses, v2_selfexpress_losses)

        self.selfexpress_losses = v1_selfexpress_losses + v2_selfexpress_losses + \
                                  v1_selfexpress_losses_pre + v2_selfexpress_losses_pre


        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses + \
                    + re_constant3 * self.coef_diff_loss

        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss)  # GradientDescentOptimizer #AdamOptimizer

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("con_Coef"))])
        # [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):

        # reader1 = tf.train.NewCheckpointReader('./models_face/model-102030-48x42-yaleb1.ckpt')
        # variables = reader1.get_variable_to_shape_map()

        all_weights = dict()
        # model_path1 = './models_mnist/view1/mnist1.ckpt'
        # model_path2 = './models_mnist/view2/mnist2.ckpt'
        model_path1 = self.pretrain_model_path1
        model_path2 = self.pretrain_model_path2

        reader1 = tf.train.NewCheckpointReader(model_path1)
        variables = reader1.get_variable_to_shape_map()
        for old_name in variables:
            all_weights[old_name] = tf.Variable(reader1.get_tensor(old_name))
        reader2 = tf.train.NewCheckpointReader(model_path2)
        variables = reader2.get_variable_to_shape_map()
        for old_name in variables:
            all_weights[old_name] = tf.Variable(reader2.get_tensor(old_name))

        data = sio.loadmat(self.init_coef_file)
        coef1 = np.array(data['coef1'])
        coef2 = np.array(data['coef2'])
        coef1 = tf.cast(coef1, tf.float32)
        coef2 = tf.cast(coef2, tf.float32)
        all_weights['coef1'] = tf.Variable(coef1, name='coef1')
        all_weights['coef2'] = tf.Variable(coef2, name='coef2')
        con_Coef = (coef1+coef2) / 2.0
        all_weights['con_Coef'] = tf.Variable(con_Coef, name='con_Coef')

        return all_weights

    def _initialize_weights1(self):

        all_weights = dict()
        # feat_size1 = self.n_input[0] * self.n_input[1]
        all_weights['v1_enc_w0'] = tf.Variable(tf.random_normal([self.feature_size[0], self.n_hidden[0]]))
        all_weights['v1_enc_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]))
        all_weights['v1_enc_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.n_hidden[1]]))
        all_weights['v1_enc_b1'] = tf.Variable(tf.random_normal([self.n_hidden[1]]))

        all_weights['v1_dec_w0'] = tf.Variable(tf.random_normal([self.n_hidden[1], self.n_hidden[0]]))
        all_weights['v1_dec_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]))
        all_weights['v1_dec_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.feature_size[0]]))
        all_weights['v1_dec_b1'] = tf.Variable(tf.random_normal([self.feature_size[0]]))

        all_weights['v2_enc_w0'] = tf.Variable(tf.random_normal([self.feature_size[1], self.n_hidden[0]]))
        all_weights['v2_enc_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]))
        all_weights['v2_enc_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.n_hidden[1]]))
        all_weights['v2_enc_b1'] = tf.Variable(tf.random_normal([self.n_hidden[1]]))

        all_weights['v2_dec_w0'] = tf.Variable(tf.random_normal([self.n_hidden[1], self.n_hidden[0]]))
        all_weights['v2_dec_b0'] = tf.Variable(tf.random_normal([self.n_hidden[0]]))
        all_weights['v2_dec_w1'] = tf.Variable(tf.random_normal([self.n_hidden[0], self.feature_size[1]]))
        all_weights['v2_dec_b1'] = tf.Variable(tf.random_normal([self.feature_size[1]]))

        data = sio.loadmat(self.init_coef_file)
        coef1 = np.array(data['coef1'])
        coef2 = np.array(data['coef2'])
        coef1 = tf.cast(coef1, tf.float32)
        coef2 = tf.cast(coef2, tf.float32)
        all_weights['coef1'] = tf.Variable(coef1, name='coef1')
        all_weights['coef2'] = tf.Variable(coef2, name='coef2')
        con_Coef = (coef1+coef2) / 2.0
        all_weights['con_Coef'] = tf.Variable(con_Coef, name='con_Coef')

        # return all_weights

        return all_weights


        ########################################################

    def encoder(self, x1, x2, weights):
        # Encoder Hidden layer with sigmoid activation #1
        v1_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x1, weights['v1_enc_w0']), weights['v1_enc_b0']))
        v1_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(v1_layer1, weights['v1_enc_w1']), weights['v1_enc_b1']))

        v2_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x2, weights['v2_enc_w0']), weights['v2_enc_b0']))
        v2_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(v2_layer1, weights['v2_enc_w1']), weights['v2_enc_b1']))
        return v1_layer2, v2_layer2

    def encoder1(self, x1, x2, weights):
        # Encoder Hidden layer with sigmoid activation #1
        v1_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x1, weights['v1_enc_w0']), weights['v1_enc_b0']))
        v1_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(v1_layer1, weights['v1_enc_w1']), weights['v1_enc_b1']))

        v2_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x2, weights['v2_enc_w0']), weights['v2_enc_b0']))
        v2_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(v2_layer1, weights['v2_enc_w1']), weights['v2_enc_b1']))
        return v1_layer1, v1_layer2, v2_layer1, v2_layer2

    # Building the decoder
    def decoder(self, v1_z, v2_z, weights):
        # Decoder Hidden layer with sigmoid activation #1
        v1_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(v1_z, weights['v1_dec_w0']), weights['v1_dec_b0']))
        v1_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(v1_layer_1, weights['v1_dec_w1']), weights['v1_dec_b1']))

        v2_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(v2_z, weights['v2_dec_w0']), weights['v2_dec_b0']))
        v2_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(v2_layer_1, weights['v2_dec_w1']), weights['v2_dec_b1']))

        return v1_layer_2, v2_layer_2

    def partial_fit(self, X1, X2, lr):  #
        cost_all, cost_reg, cost_recons, cost_selfexprs, summary, _, Coef = \
            self.sess.run((self.loss, self.reg_losses,
                           self.reconst_cost, self.selfexpress_losses,
                           self.merged_summary_op, self.optimizer, self.con_Coef),
                          feed_dict={self.x1: X1, self.x2: X2, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost_all, cost_reg, cost_recons, cost_selfexprs, Coef

    def initlization(self):
        self.sess.run(self.init)


def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1

    uu, ss, _ = svd(C)
    UU = uu[:, :r]
    SS = ss[:r]

    U, S, _ = svds(C, k=r, v0=np.ones(C.shape[0]))

    U = U[:, ::-1]
    S = np.sqrt(S[::-1])

    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C, axis=0)
    W = np.diag(1.0 / W)
    L = W.dot(C)
    return L


def test_face(Img, Label, CAE, num_class):
    alpha = max(0.4 - (num_class - 1) / 10 * 0.1, 0.1)
    print alpha

    acc_ = []
    for i in range(0, 39 - num_class):
        face_10_subjs = np.array(Img[64 * i:64 * (i + num_class), :])
        face_10_subjs = face_10_subjs.astype(float)
        label_10_subjs = np.array(Label[64 * i:64 * (i + num_class)])
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs)

        CAE.initlization()
        CAE.restore()  # restore from pre-trained model

        max_step = 50 + num_class * 25  # 100+num_class*20
        display_step = max_step
        lr = 1.0e-3
        # fine-tune network
        epoch = 0
        while epoch < max_step:
            epoch = epoch + 1
            cost, Coef = CAE.partial_fit(face_10_subjs, lr)  #
            if epoch % display_step == 0:
                print "epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size))
                Coef = thrC(Coef, alpha)
                y_x, _ = post_proC(Coef, label_10_subjs.max(), 10, 3.5)
                missrate_x = err_rate(label_10_subjs, y_x)
                acc_x = 1 - missrate_x
                print "experiment: %d" % i, "our accuracy: %.4f" % acc_x
        acc_.append(acc_x)

    acc_ = np.array(acc_)
    m = np.mean(acc_)
    me = np.median(acc_)
    print("%d subjects:" % num_class)
    print("Mean: %.4f%%" % ((1 - m) * 100))
    print("Median: %.4f%%" % ((1 - me) * 100))
    print(acc_)

    return (1 - m), (1 - me)



def train_face(Img, CAE, batch_size):
    it = 0
    display_step = 500
    save_step = 5000
    _index_in_epoch = 0
    _epochs = 0

    max_num = 80001
    while it < max_num:
        batch_x, _index_in_epoch, _epochs = next_batch(Img, _index_in_epoch, batch_size, _epochs)
        batch_x = np.reshape(batch_x, [batch_size, Img.shape[1]])
        cost = CAE.partial_fit(batch_x)
        it = it + 1
        avg_cost = cost / (batch_size)
        if it % display_step == 0:
            print  ("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model()
    return


def test_face_multi(img1, img2, Label, DMVSC, lr, alpha, post_param1, post_param2, coef_save_path):
    DMVSC.initlization()
    display_step = 10
    max_step = display_step * 600  # 100+num_class*20
    
    epoch = 0
    batch_size = img1.shape[0]
    coefs = []
    accs = []
    clabels = []
    while epoch < max_step:
        epoch = epoch + 1
        cost_all, cost_reg, cost_recons, cost_selfexprs, Coef = DMVSC.partial_fit(img1, img2, lr)  #
        if epoch % display_step == 0:
            Coef = thrC(Coef, alpha)
            Coef = np.abs(Coef)
            y_x, _ = post_proC(Coef, Label.max(), post_param1, post_param2)
            missrate_x = err_rate(Label, y_x)
            acc_x = 1 - missrate_x
            print "cost_all: %.8f" % (cost_all / float(batch_size)), "cost_recons: %.8f" % (
            cost_recons / float(batch_size)), \
                "our accuracy: %.4f" % acc_x
            accs.append(acc_x)
            clabels.append(y_x)
    max_idx = np.argmax(np.array(accs))
    max_acc = accs[max_idx]
    print("max_acc: %.8f" % max_acc)
    plabel = clabels[max_idx]
    flabel = y_x
    sio.savemat(coef_save_path, dict(plabel=plabel, flabel=flabel, gt=Label))


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-is_pretrain', type=bool, default=False)
    # parser.add_argument('-n_hidden', type=list, default=[128, 128])
    parser.add_argument('-n_hidden1', type=int, default=128)
    parser.add_argument('-n_hidden2', type=int, default=128)
    parser.add_argument('-reg1', type=float, default=0.01)
    parser.add_argument('-reg2', type=float, default=0.01)
    parser.add_argument('-reg3', type=float, default=0.01)
    parser.add_argument('-lr', type=float, default=1.0e-3)
    parser.add_argument('-alpha', type=float, default=0.4)
    parser.add_argument('-post_param1', type=int, default=3)
    parser.add_argument('-post_param2', type=int, default=1)
    return parser


if __name__ == '__main__':

    parser = args_parser()    
    args = parser.parse_args()

    reg1 = args.reg1
    reg2 = args.reg2
    reg3  = args.reg3
    # n_hidden = args.n_hidden
    n_hidden = [args.n_hidden1, args.n_hidden2]
    lr = args.lr
    alpha = args.alpha
    post_param1 = args.post_param1
    post_param2 = args.post_param2
    is_pretrain = args.is_pretrain

    pretrain1 = False
    pretrain2 = False
    if is_pretrain:
        pretrain1 = True
        pretrain2 = True
    

    model_path1 = './bbc_sport/view1/view1.ckpt'
    model_path2 = './bbc_sport/view2/view2.ckpt'

    model_path = './bbc_sport/bbc_sport.ckpt'
    restore_path = './bbc_sport/bbc_sport.ckpt'
    logs_path = './logs'

    init_coef_file = './bbc_sport/bbcsport_2view_norm_data_ssc_init.mat'
    result_path = './bbc_sport/bbc_sport_result_without_init.mat'

    data = sio.loadmat(init_coef_file)

    X1 = data['X1']
    X2 = data['X2']    
    gt = data['gt']
    num_sample = gt.shape[0]
    batch_size = num_sample
    feat_size = np.array([X1.shape[0], X2.shape[0]])
    I1 = np.array(X1)
    I2 = np.array(X2)

    Label = np.array(gt[:])
    Label = Label.flatten()
    pretrain_batchsize = int(num_sample / 5.0)
    I1 = np.transpose(I1)
    I2 = np.transpose(I2)


    if pretrain1:
    	tf.reset_default_graph()
        CAE1 = ConvAE1(feature_size=feat_size[0], n_hidden=n_hidden, learning_rate=lr, batch_size=pretrain_batchsize, \
                       model_path=model_path1, restore_path=model_path1)
        train_face(I1, CAE1, pretrain_batchsize)

    if pretrain2:
    	tf.reset_default_graph()
        CAE2 = ConvAE2(feature_size=feat_size[1], n_hidden=n_hidden, learning_rate=lr, batch_size=pretrain_batchsize, \
                       model_path=model_path2, restore_path=model_path2)
        train_face(I2, CAE2, pretrain_batchsize)

    num_class = np.max(Label)
    
    tf.reset_default_graph()
    DMVSC = dmvsc(feature_size=feat_size, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, re_constant3=reg3,
                  batch_size=batch_size,  model_path=model_path, restore_path=restore_path, logs_path=logs_path,
                  pretrain_model_path1=model_path1, pretrain_model_path2=model_path2, init_coef_file=init_coef_file)
    test_face_multi(I1, I2, Label, DMVSC, lr, alpha, post_param1, post_param2, result_path)


    print  "params config: "
    print  "reg1: %.2f" % reg1
    print  "reg2: %.2f" % reg2
    print  "reg3: %.2f" % reg3
    print  "n_hidden 1: %d" % n_hidden[0]
    print  "n_hidden 2: %d" % n_hidden[1]
    print  "lr: %.5f" % lr
    print  "alpha: %.2f" % alpha
    print  "post_param1: %d" % post_param1
    print  "post_param2: %d" % post_param2
    print "===================================================="    
